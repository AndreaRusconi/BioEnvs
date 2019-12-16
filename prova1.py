from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

tf.keras.backend.clear_session()  # For easy reset of notebook state.

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

new_model = np.load('modello/parameterss/model/values_fn/qf2/fc1/bias:0.npy')
print(len(new_model))
print(new_model[0])