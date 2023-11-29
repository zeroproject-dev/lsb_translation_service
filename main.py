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

from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

actions = np.array(["hola", "gracias", "te amo"])
nn = GRU
model = Sequential()
model.add(
    Bidirectional(
        nn(64, return_sequences=True, activation="relu", input_shape=(30, 1662))
    )
)
model.add(Bidirectional(nn(128, return_sequences=True, activation="relu")))
model.add(Bidirectional(nn(64, return_sequences=False, activation="relu")))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(actions.shape[0], activation="softmax"))

dummy_input = np.random.rand(1, 30, 1662).astype(np.float32)
_ = model(dummy_input)

model.load_weights("lsb_B_GRU.h5")
_ = model.predict(dummy_input)


holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

user_data_channels = {}


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, user_id):
        super().__init__()
        self.track = track

        self.user_id = user_id

        self.sequence = []
        self.predictions = []
        self.threshold = 0.7
        self.prev = ""

        self.pose_dimensions = 33 * 4
        self.face_dimensions = 468 * 3
        self.hand_dimensions = 21 * 3
        self.keypoints = np.zeros(
            self.pose_dimensions + self.face_dimensions + 2 * self.hand_dimensions
        )

    async def recv(self):
        frame = await self.track.recv()
        results = mediapipe_detection(frame.to_ndarray(format="bgr24"), holistic)

        keypoints = extract_keypoints(results)
        self.sequence.append(keypoints)

        if len(self.sequence) == 30:
            asyncio.ensure_future(self.process_predictions())

        return frame

    async def process_predictions(self):
        res = model.predict(np.expand_dims(self.sequence, axis=0))[0]
        prediction = np.argmax(res)

        if res[prediction] > self.threshold:
            if actions[prediction] != self.prev:
                self.prev = actions[prediction]
                user_data_channels[self.user_id].send(
                    json.dumps({"prediction": actions[prediction]})
                )

        self.sequence.clear()


async def offer(request):
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
            # vtt.data_channel = user_data_channels.get(pc_id)
            print("TRACK")
            print(pc_id)
            pc.addTrack(vtt)

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    if answer is not None:
        await pc.setLocalDescription(answer)

    res = json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

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
    app = web.Application()
    cors = aiohttp_cors.setup(app)
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/offer", offer)
    # app.router.add_post("/update", update_nn)

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

    web.run_app(app, access_log=None, host="0.0.0.0", port=6969)

    if holistic:
        holistic.close()
        holistic = None
