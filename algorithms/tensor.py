import tensorflow as tf
import numpy as np
from algorithms.norms import lerp


def vals_to_space(x, weights, min_val=0.0, max_val=1.0, out_shape=tf.constant([10, 10, 10])):
    x = (x - min_val) / (max_val - min_val)
    n = tf.cast(out_shape - tf.ones_like(out_shape), tf.float32)

    x = x * n
    a = tf.floor(x)
    b = tf.ceil(x)

    a_f = tf.reduce_sum(tf.square(x - a), axis=-1)
    b_f = tf.reduce_sum(tf.square(x - b), axis=-1)

    a_f = tf.where(tf.equal(a_f, 0.0), tf.ones_like(a_f), a_f)
    b_f = tf.where(tf.equal(b_f, 0.0), tf.ones_like(b_f), b_f)

    a_v = a_f / tf.maximum(a_f, b_f) * weights
    b_v = b_f / tf.maximum(a_f, b_f) * weights

    indices = tf.concat((a, b), axis=0)
    indices = tf.cast(indices, tf.int32)
    updates = tf.concat((a_v, b_v), axis=0)

    out = tf.scatter_nd(indices, updates, out_shape)
    out = tf.clip_by_value(out, 0.0, 1.0)
    return out


def space_to_maps(x, reduction=tf.reduce_max):
    yz = tf.expand_dims(reduction(x, axis=1), axis=-1)
    xz = tf.expand_dims(reduction(x, axis=2), axis=-1)
    xy = tf.expand_dims(reduction(x, axis=3), axis=-1)

    return tf.concat((yz, xz, xy), axis=-1)


def maps_to_points(x, min_val=0.7):
    yz, xz, xy = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    h, w = np.shape(xy)[:2]
    d = h
    xy = np.reshape(np.tile(xy, h), (h, w, d))
    xz = np.reshape(np.tile(xz, h), (h, w, d)).transpose([0, 2, 1])
    yz = np.reshape(np.tile(yz, h), (h, w, d)).transpose([1, 2, 0])
    space = np.where(np.minimum(np.minimum(yz, xz), xy) >= min_val, (xy + xz + yz) / 3, 0.0)
    space = np.where(space >= min_val, space, 0.0)

    points = list()
    weights = list()

    for i in range(h):
        for j in range(w):
            for k in range(d):
                if space[i, j, k] != 0.0:
                    points.append([float(i) / h, float(j) / w, float(k) / d])
                    weights.append(space[i, j, k])

    return points, weights


def  space_to_points(tensor, min_val=0.0, max_val=1.0):
    x, y, z = np.nonzero(tensor)
    shape = np.shape(tensor)

    x = lerp(x, 0.0, float(shape[2] - 1), min_val, max_val)
    y = lerp(y, 0.0, float(shape[1] - 1), min_val, max_val)
    z = lerp(z, 0.0, float(shape[0] - 1), min_val, max_val)

    return [x, y, z]


def gaussian_kernel(mean=0.0, std=1.0, size=(5, 5, 5), norm='sum'):
    x = np.zeros(size)

    d = tf.distributions.Normal(mean, std)
    mid = np.array(size, dtype=np.float32) / 2 - 0.5

    indices = []
    for i, axis_size in enumerate(size):
        if len(indices) == 0:
            for j in range(axis_size):
                indices.append([j])
        else:
            prev_indices = indices
            indices = list()
            for idx in prev_indices:
                for j in range(axis_size):
                    indices.append(idx + [j])

    indices = np.array(indices)

    for idx in indices:
        x[tuple(idx)] = np.sqrt(np.sum(np.square(mid - np.array(idx, dtype=np.float32))))

    k = d.prob(x)

    if norm == 'sum':
        return k / tf.reduce_sum(k)
    elif norm == 'max':
        return k / tf.reduce_max(k)
    else:
        return k


def conv_kernel_3d(x, kernel, strides=(1, 1, 1, 1, 1)):
    return tf.nn.conv3d(x, filter=kernel, strides=strides, padding='SAME')


def conv_kernel_2d(x, kernel, strides=(1, 1, 1, 1)):
    return tf.nn.conv2d(x, filter=kernel, strides=strides, padding='SAME')


def conv_kernel_1d(x, kernel, strides=(1, 1, 1)):
    return tf.nn.conv1d(x, filter=kernel, strides=strides, padding='SAME')
