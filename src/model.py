from keras.layers import Dense, GRU, Bidirectional, Input
from keras.models import Sequential
import numpy as np
import requests


def create_model(actions):
  nn = GRU
  model = Sequential()
  model.add(Input(shape=(30, 1662)))
  model.add(Bidirectional(nn(64, return_sequences=True, activation="relu")))
  model.add(Bidirectional(nn(128, return_sequences=True, activation="relu")))
  model.add(Bidirectional(nn(64, return_sequences=False, activation="relu")))
  model.add(Dense(64, activation="relu"))
  model.add(Dense(32, activation="relu"))
  model.add(Dense(actions.shape[0], activation="softmax"))

  dummy_input = np.random.rand(1, 30, 1662).astype(np.float32)
  _ = model(dummy_input)

  return model


def instance_model(weight_path):
  api_url = "http://localhost:3300/api/v1/words"
  actions = requests.get(api_url, params={'status': 'active'}).json()
  actions = np.array([action['word'] for action in actions['data']])

  model = create_model(actions)
  model.load_weights(weight_path)
  model.predict(np.random.rand(1, 30, 1662).astype(np.float32))
  return model, actions
