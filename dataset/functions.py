import tensorflow as tf
import numpy as np
import os
import re


def decode_image(x, channels=3, dtype=tf.float32):
    image_string = tf.read_file(x)
    image_decoded = tf.image.decode_image(image_string, channels=channels)
    image = tf.cast(image_decoded, dtype)
    return image


def random_jitter(x, max_scale):
    shape = tf.shape(x)

    scale = tf.random.uniform([], 1.0, max_scale)

    size = tf.cast(tf.cast(shape[1:3], dtype=tf.float32) * scale, dtype=tf.int32)

    x = tf.image.resize(x, size)

    return tf.image.random_crop(x, shape)


def get_nested_files(path):
    paths = []
    for dir, _, files in os.walk(path):
        for file in files:
            paths.append(os.path.join(dir, file))

    return paths


def filter_by_endings(x, endings):
    pattern = '.+({0})$'.format('|'.join(endings))
    regex = re.compile(pattern)
    x = list(filter(regex.match, x))
    return x


def split(x, *args):
    if len(args) < 2:
        return [x]

    sum = 0
    for num in args:
        sum += float(num)

    chunks = []

    weights = np.array(args)
    weights /= np.linalg.norm(weights, ord=1)
    idx = np.cumsum(weights) * len(x)
    idx = np.insert(idx.astype(np.int32), 0, 0)

    for i in range(len(idx) - 1):
        chunks.append(x[idx[i]:idx[i + 1]])

    return chunks


def set_shape_and_return(x, shape):
    x.set_shape(shape)
    return x
