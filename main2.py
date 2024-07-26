from src.model import create_model
from keras.utils import plot_model
import numpy as np

model = create_model(np.array(['hola', 'adiós', 'test']))

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True,
           expand_nested=True, show_layer_activations=True, dpi=96)
