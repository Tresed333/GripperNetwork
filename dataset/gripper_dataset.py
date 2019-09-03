import tensorflow as tf
from dataset.functions import *
from algorithms.norms import lerp
from algorithms.geometry import log_map
import os

TRAIN_FOLDER = 'train'
VALID_FOLDER = 'val'
TEST_FOLDER = 'test'


def get(dataset_path, batch_size, resize_dims=None, map_range=None):
    train_dataset = _prepare_dataset(dataset_path, TRAIN_FOLDER, batch_size, resize_dims, map_range)
    val_dataset = _prepare_dataset(dataset_path, VALID_FOLDER, batch_size, resize_dims, map_range)
    test_dataset = _prepare_dataset(dataset_path, TEST_FOLDER, 1, resize_dims, map_range=map_range)

    return train_dataset, val_dataset, test_dataset


def _prepare_dataset(path, folder, batch_size, resize_dims=None, map_range=None):
    def _decode_image(image):
        image_string = tf.read_file(image)
        image_decoded = tf.image.decode_png(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32)

        return image

    p = os.path.join(path, folder)

    if not os.path.exists(p):
        return None

    rgb, depth, translation, rotation = _load_data(p)

    rgb = tf.convert_to_tensor(rgb, dtype=tf.string)
    depth = tf.convert_to_tensor(depth, dtype=tf.string)

    ds = tf.data.Dataset.from_tensor_slices((rgb, depth, translation, rotation))
    ds = ds.shuffle(len(rgb)).map(lambda x, y, t, r: [_decode_image(x), _decode_image(y), t, r])
    if resize_dims is not None:
        ds = ds.map(
            lambda x, y, t, r: [tf.image.resize_images(x, resize_dims), tf.image.resize_images(y, resize_dims), t, r])

    if map_range is not None:
        ds = ds.map(lambda x, y, t, r: [lerp(x, *map_range), lerp(y, *map_range), t, r])

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

    rgb = _load_pictures(os.path.join(path, "rgb"))
    depth = _load_pictures(os.path.join(path, "depth"))
    trans, rot = _load_translations(os.path.join(path, "trans"))
    return [rgb, depth, trans, rot]


def process(data):
    rgb, depth, trans, rot = data

    log = log_map(rot)

    params = tf.concat((trans, log), axis=-1)
    return rgb, depth, trans, rot, params


def dictify(data):
    rgb, depth, trans, rot, params = data
    return {
        'rgb': rgb,
        'depth': depth,
        't': trans,
        'r': rot,
        'params': params
    }
