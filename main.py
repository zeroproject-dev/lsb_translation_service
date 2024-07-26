import threading
import queue
import asyncio
import json
import logging
import os
import uuid
import aiohttp_cors

from keras.models import Sequential
from keras.layers import Dense, GRU, Bidirectional
import numpy as np

from src.detection import extract_keypoints, mediapipe_detection, mp_holistic

from aiohttp import web

from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.contrib.media import MediaRelay

from src.model import instance_model


MODEL_FILE_NAME = "lsb.keras"
ROOT = os.path.dirname(__file__)
WEIGHTS = os.path.join(ROOT, MODEL_FILE_NAME)


model, actions = instance_model(weight_path=WEIGHTS)


async def save_nn(request: web.Request):
  reader = await request.multipart()

  field = await reader.next()

  size = 0
  with open(os.path.join(MODEL_FILE_NAME), 'wb') as f:
    while True:
      chunk = await field.read_chunk()  # 8192 bytes by default.
      if not chunk:
        break
      size += len(chunk)
      f.write(chunk)


async def update_nn(request: web.Request):
  await save_nn(request)

  global model, actions
  model, actions = instance_model(weight_path=WEIGHTS)

  return web.Response(text="OK")


holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()


user_data_channels = {}


class FrameProcessor(threading.Thread):
  def __init__(self, user_id, model, actions, threshold, result_queue):
    super().__init__()
    self.queue = queue.Queue()
    self.user_id = user_id
    self.model = model
    self.actions = actions
    self.threshold = threshold
    self.sequence = []
    self.predictions = []
    self.prev = ""
    self.pose_dimensions = 33 * 4
    self.face_dimensions = 468 * 3
    self.hand_dimensions = 21 * 3
    self.keypoints = np.zeros(
        self.pose_dimensions + self.face_dimensions + 2 * self.hand_dimensions
    )
    self.result_queue = result_queue
    self.start()

  def run(self):
    while True:
      frame = self.queue.get()
      if frame is None:
        break
      self.process_predictions(frame)
      self.queue.task_done()

  def process_predictions(self, frame):
    results = mediapipe_detection(
        frame.to_ndarray(format="bgr24"), holistic)
    keypoints = extract_keypoints(results)
    self.sequence.append(keypoints)
    if len(self.sequence) >= 30:
      res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
      prediction = np.argmax(res)
      print("PREDICTION", res, prediction, self.actions[prediction])
      if res[prediction] > self.threshold:
        if self.actions[prediction] != self.prev:
          self.prev = self.actions[prediction]
          self.result_queue.put(
              (self.user_id, self.actions[prediction]))
      self.sequence.clear()

  def process_frame(self, frame):
    self.queue.put(frame)

  def stop(self):
    self.queue.put(None)
    self.join()


class VideoTransformTrack(MediaStreamTrack):
  """
  A video stream track that transforms frames from an another track.
  """

  kind = "video"

  def __init__(self, track, user_id):
    super().__init__()
    self.track = track

    self.user_id = user_id

    self.threshold = 0.7

    self.result_queue = queue.Queue()
    self.frame_processor = FrameProcessor(
        user_id, model, actions, self.threshold, self.result_queue
    )

  async def recv(self):
    frame = await self.track.recv()
    self.frame_processor.process_frame(frame)
    while not self.result_queue.empty():
      user_id, prediction = self.result_queue.get()
      if user_id == self.user_id:
        print("SENDING", prediction)
        user_data_channels[user_id].send(
            json.dumps({"prediction": prediction}))
      self.result_queue.task_done()
    return frame

  def stop(self) -> None:
    self.frame_processor.stop()
    self.result_queue = queue.Queue()
    super().stop()


async def offer(request: web.Request):
  params = await request.json()
  offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

  config = RTCConfiguration(
      [
          RTCIceServer("stun:stun.l.google.com:19302"),
      ]
  )

  pc = RTCPeerConnection(config)
  pc_id = "PeerConnection(%s)" % uuid.uuid4()
  pcs.add(pc)

  def log_info(msg, *args):
    logger.info(pc_id + " " + msg, *args)

  print("Created for %s", request.remote)

  @pc.on("datachannel")
  def on_datachannel(channel):
    print("ON DATACHANNEL")
    print(pc_id)
    print(channel)
    user_data_channels[pc_id] = channel

  @pc.on("connectionstatechange")
  async def on_connectionstatechange():
    log_info("Connection state is %s", pc.connectionState)
    if pc.connectionState == "failed":
      await pc.close()
      pcs.discard(pc)

  @pc.on("track")
  def on_track(track):
    log_info("Track %s received", track.kind)

    if track.kind == "video":
      vtt = VideoTransformTrack(relay.subscribe(track), pc_id)
      print("TRACK")
      print(pc_id)
      pc.addTrack(vtt)

    @track.on("ended")
    async def on_ended():
      vtt.stop()
      log_info("Track %s ended", track.kind)

  # handle offer
  await pc.setRemoteDescription(offer)

  # send answer
  answer = await pc.createAnswer()
  if answer is not None:
    await pc.setLocalDescription(answer)

  res = json.dumps({"sdp": pc.localDescription.sdp,
                   "type": pc.localDescription.type})

  return web.Response(
      content_type="application/json",
      text=res,
  )


async def on_shutdown(app):
  # close peer connections
  coros = [pc.close() for pc in pcs]
  await asyncio.gather(*coros)
  pcs.clear()


if __name__ == "__main__":
  app = web.Application(client_max_size=1024 ** 4)
  cors = aiohttp_cors.setup(app)
  app.on_shutdown.append(on_shutdown)
  app.router.add_post("/offer", offer)
  app.router.add_post("/update", update_nn)

  for route in list(app.router.routes()):
    cors.add(
        route,
        {
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*",
            )
        },
    )

  web.run_app(app, host="0.0.0.0", port=6969)

  if holistic:
    holistic.close()
    holistic = None
