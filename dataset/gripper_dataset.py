import sys

import tensorflow as tf

from algorithms.image import coordinates
from algorithms.tensor import gaussian_kernel
from dataset.functions import *
from algorithms.norms import lerp
from algorithms.geometry import log_map
import os
import numpy as np

TRAIN_FOLDER = 'train'
VALID_FOLDER = 'val'
TEST_FOLDER = 'test'
240


def get(dataset_path, batch_size, resize_dims=None, map_range=None, kernel_size=(200, 200)):
    train_dataset = _prepare_dataset(dataset_path, TRAIN_FOLDER, batch_size, resize_dims, map_range, kernel_size)
    val_dataset = _prepare_dataset(dataset_path, VALID_FOLDER, batch_size, resize_dims, map_range, kernel_size)
    test_dataset = _prepare_dataset(dataset_path, TEST_FOLDER, 1, resize_dims, map_range, kernel_size)

    return train_dataset, val_dataset, test_dataset


def _prepare_dataset(path, folder, batch_size, resize_dims=None, map_range=None, kernel_size=None):
    def _decode_image(image, channels=3):
        image_string = tf.read_file(image)
        image_decoded = tf.image.decode_png(image_string, channels=channels)
        image = tf.cast(image_decoded, tf.float32)

        return image

    def _box_to_map(box, kernel, image_shape):
        tl = tf.cast(box[:2], tf.float32)
        br = tf.cast(box[2:], tf.float32)

        center = (tl + br) / 2

        tl_in_image = center - tf.cast(tf.shape(kernel), tf.float32) / 2
        br_in_image = tl_in_image + tf.cast(tf.shape(kernel), tf.float32)

        zeros = tf.zeros([2])
        image_size = image_shape[:2]
        image_size_reversed = tf.reverse(image_size, axis=[0])

        tl_fit = tf.maximum(tl_in_image, zeros)
        br_fit = tf.minimum(br_in_image, tf.cast(image_size_reversed, tf.float32))

        begin_cut = tf.abs(tl_fit - tl_in_image)
        end_cut = tf.abs(br_in_image - br_fit)

        image_begin = tl_fit
        image_end = br_fit

        begin = zeros + begin_cut
        end = tf.cast(tf.shape(kernel), tf.float32) - end_cut

        image_begin = tf.cast(image_begin, tf.int32)
        image_end = tf.cast(image_end, tf.int32)
        begin = tf.cast(begin, tf.int32)
        end = tf.cast(end, tf.int32)

        window_disparity = (image_end - image_begin) - (end - begin)
        kernel_slice = tf.slice(kernel, begin, end - begin + window_disparity)

        updates = tf.reshape(kernel_slice, shape=[-1])

        tl_fit_inv = tf.reverse(tl_fit, axis=[0])
        indices = tf.reshape(coordinates(end - begin + window_disparity), shape=(-1, 2)) + tf.cast(
            tl_fit_inv, tf.int32)

        object_map = tf.scatter_nd(indices, updates, image_size)
        return tf.expand_dims(object_map, axis=-1)

    kernel = gaussian_kernel(std=kernel_size[0] / 5.0, size=kernel_size, norm='max')
    p = os.path.join(path, folder)

    if not os.path.exists(p):
        return None

    rgb, depth, translation, rotation, boxes = _load_data(p)

    rgb = tf.convert_to_tensor(rgb, dtype=tf.string)
    depth = tf.convert_to_tensor(depth, dtype=tf.string)

    ds = tf.data.Dataset.from_tensor_slices((rgb, depth, translation, rotation, boxes))
    ds = ds.map(lambda x, y, t, r, b: [_decode_image(x), _decode_image(y, 1), t, r, b])
    ds = ds.map(lambda x, y, t, r, b: [x, y, t, r, _box_to_map(b, kernel, tf.shape(x)), b])

    if resize_dims is not None:
        ds = ds.map(
            lambda x, y, t, r, m, b: [tf.image.resize_images(x, resize_dims), tf.image.resize_images(y, resize_dims), t,
                                      r,
                                      tf.image.resize_images(m, resize_dims), b])

    if map_range is not None:
        ds = ds.map(lambda x, y, t, r, m, b: [lerp(x, *map_range), lerp(y, *map_range), t, r, m, b])

    ds = ds.batch(batch_size).prefetch(batch_size)

    return ds


def _load_data(path):
    def _load_pictures(path):
        toReturn = list()
        for file in sorted(os.listdir(path)):
            toReturn.append(os.path.join(path, file))
        return toReturn

    def _load_translations(path):
        returnTrans = list()
        returnRot = list()
        for translation in sorted(os.listdir(path)):
            filePath = os.path.join(path, translation)
            b = np.loadtxt(filePath, dtype=float)
            rot = b[:3, :3]
            trans = b[:3, 3]
            trans = np.reshape(trans, newshape=(-1))
            returnTrans.append(trans)
            returnRot.append(rot)

        return returnTrans, returnRot

    def _load_bbox(path):
        def _refine(box):
            x1 = box[0]
            x2 = box[2]
            y1 = box[1]
            y2 = box[3]

            left = x1 if x1 < x2 else x2
            right = x2 if x1 < x2 else x1
            top = y1 if y1 < y2 else y2
            bottom = y2 if y1 < y2 else y1

            return np.array([left, top, right, bottom])

        returnBbox = list()
        for bbox in sorted(os.listdir(path)):
            final_path = os.path.join(path, bbox)
            box = np.loadtxt(final_path)
            box = np.reshape(box, (-1))
            box = _refine(box)
            returnBbox.append(box)
        return returnBbox

    rgb = _load_pictures(os.path.join(path, "rgb"))
    depth = _load_pictures(os.path.join(path, "depth"))
    trans, rot = _load_translations(os.path.join(path, "trans"))
    bbox = _load_bbox(os.path.join(path, 'bbox'))
    return [rgb, depth, trans, rot, bbox]


def process(data):
    rgb, depth, trans, rot, object_map, box = data

    log = log_map(rot)

    params = tf.concat((trans, log), axis=-1)
    return rgb, depth, trans, rot, params, object_map, box


def dictify(data):
    rgb, depth, trans, rot, params, object_map, box = data
    return {
        'rgb': rgb,
        'depth': depth,
        't': trans,
        'r': rot,
        'params': params,
        'map': object_map,
        'box': box
    }
