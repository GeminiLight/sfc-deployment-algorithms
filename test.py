# import numpy as np

# # a = np.array([True, True, False, True])
# # a[[1,2]]=False
# # print(a)


# a = np.array([True, False, True, True, False, True])
# b = np.array([False, True, True, True, False, True])
# c = np.array([True, True, False, True, True, False])
# # print(np.logical_and(np.logical_and(a, b), c))

# import os
# import sys
# import time

# file_path_dir = os.path.abspath('.')
# if os.path.abspath('.') not in sys.path:
#     sys.path.append(file_path_dir)


# import numpy as np
# import spektral as sk
# import tensorflow as tf
# import tensorflow_probability as tfp
# from config import get_config
# from spektral.layers import GraphConv
# from tensorflow.keras import Model
# from tensorflow.keras.layers import (GRU, BatchNormalization, Conv1D, Dense,
#                                      Dropout, Embedding, Flatten, Layer,
#                                      Softmax)
# from tensorflow.keras.losses import (MeanSquaredError,
#                                      SparseCategoricalCrossentropy)
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.regularizers import l2
# sample_output = np.array([0.0, 0.0, 0.042857286, 0.011323607, 0.06521806, 0.083010435, 0.0, 0.03522057, 0.0, 0.0, 0.0, 0.007439192, 0.0, 0.015633563])

# sm = Softmax()

# print(sm(sample_output))
# dist = tfp.distributions.Categorical(probs=sample_output, dtype=tf.float32)
# action = dist.sample(sample_shape=(100))
# print(action.numpy())
# sample_output = np.array([0.0, 0.0])
# print(tf.argmax(sample_output))

import tensorflow as tf
import tensorflow_probability as tfp

p = tf.constant([0.1, 0.2, 0.3, 0.4])
p = p.numpy()

p[p==0.2] = 0.1
print(p)


