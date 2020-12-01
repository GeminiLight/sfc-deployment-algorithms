import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

'''mask'''
# y_true = [1, 1, 0, 1]
# mask = tf.math.logical_not(tf.math.equal(y_true, 0)) 
# print(mask)
# mask = tf.cast(mask, dtype=tf.int32)
# print(mask)

'''sce'''
# y_true = [[1, 2], [2, 2]]
# y_pred = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]], [[0.1, 0.8, 0.1], [0.1, 0.8, 0.1]]]
# # Using 'auto'/'sum_over_batch_size' reduction type.
# scce = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')
# a = scce(y_true, y_pred, sample_weight=tf.constant([0.3, 0.7]))
# print(a)

a = [[1, 2, 3, 4], [5, 6, 7, 8]]
print()
print(tf.stack(a, axis=1))