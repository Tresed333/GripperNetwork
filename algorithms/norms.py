import tensorflow as tf


def min_max(x):
    return (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))


def lerp(x, x1, x2, y1, y2):
    # Figure out how 'wide' each range is
    x_span = x2 - x1
    y_span = y2 - y1

    # Convert the left range into a 0-1 range (float)
    scaled = (x - x1) / x_span

    # Convert the 0-1 range into a value in the right range.
    return y1 + (scaled * y_span)
