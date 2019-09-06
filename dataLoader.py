import tensorflow as tf
tf.enable_eager_execution()

import matplotlib.pyplot as plt

tl_fit = tf.constant([0.0, 1.0], dtype=tf.float32)

tl_fit_inv = tf.reverse(tl_fit, axis=[0])

print(tl_fit_inv)
