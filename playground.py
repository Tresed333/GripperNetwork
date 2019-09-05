import tensorflow as tf

tf.enable_eager_execution()

import matplotlib.pyplot as plt

from algorithms.image import coordinates
from algorithms.tensor import gaussian_kernel


def _box_to_map(box, kernel, image_shape):
    tl = tf.cast(box[:2], tf.float32)
    br = tf.cast(box[2:], tf.float32)

    center = (tl + br) / 2

    tl_in_image = center - tf.cast(tf.shape(kernel), tf.float32) / 2
    br_in_image = tl_in_image + tf.cast(tf.shape(kernel), tf.float32)

    zeros = tf.zeros([2])
    image_size = image_shape[:2]

    tl_fit = tf.maximum(tl_in_image, zeros)
    br_fit = tf.minimum(br_in_image, tf.cast(image_size, tf.float32))

    begin_cut = tl_fit - tl_in_image
    end_cut = br_in_image - br_fit

    image_begin = center - tf.cast(tf.shape(kernel), tf.float32) / 2 + begin_cut
    image_end = center + tf.cast(tf.shape(kernel), tf.float32) / 2 - end_cut

    begin = zeros + begin_cut
    end = tf.cast(tf.shape(kernel), tf.float32) - end_cut

    image_begin = tf.cast(image_begin, tf.int32)
    image_end = tf.cast(image_end, tf.int32)
    begin = tf.cast(begin, tf.int32)
    end = tf.cast(end, tf.int32)

    window_disparity = (image_end - image_begin) - (end - begin)

    kernel_slice = tf.slice(kernel, begin, end - begin + window_disparity)

    updates = tf.reshape(kernel_slice, shape=[-1])
    indices = tf.reshape(coordinates(end + window_disparity), shape=(-1, 2)) + tf.cast(center, tf.int32)

    object_map = tf.scatter_nd(indices, updates, image_size)

    return object_map


box = tf.constant([400, 400, 500, 500], dtype=tf.float32)
kernel_size = [100, 100]
kernel = gaussian_kernel(std=20.0, size=kernel_size, norm='max')
image_shape = tf.constant([500, 500, 3], dtype=tf.int32)

maps = _box_to_map(box, kernel, image_shape)
plt.imshow(maps.numpy())
plt.show()
pass

#
# def tridiagonal(diag, sub, sup):
#     n = tf.shape(diag)[0]
#     r = tf.range(n)
#     ii = tf.concat([r, r[1:], r[:-1]], axis=0)
#     jj = tf.concat([r, r[:-1], r[1:]], axis=0)
#     idx = tf.stack([ii, jj], axis=1)
#     values = tf.concat([diag, sub, sup], axis=0)
#     return tf.scatter_nd(idx, values, [n, n])
#
#
# out = tridiagonal([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
# print(out)
