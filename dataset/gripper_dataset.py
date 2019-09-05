import tensorflow as tf

from algorithms.tensor import gaussian_kernel
from dataset.functions import *
from algorithms.norms import lerp
from algorithms.geometry import log_map
import os

TRAIN_FOLDER = 'train'
VALID_FOLDER = 'val'
TEST_FOLDER = 'test'


def get(dataset_path, batch_size, resize_dims=None, image_size=None, map_range=None):
    train_dataset = _prepare_dataset(dataset_path, TRAIN_FOLDER, batch_size, resize_dims, image_size, map_range)
    val_dataset = _prepare_dataset(dataset_path, VALID_FOLDER, batch_size, resize_dims, image_size, map_range)
    test_dataset = _prepare_dataset(dataset_path, TEST_FOLDER, 1, resize_dims, image_size, map_range=map_range)

    return train_dataset, val_dataset, test_dataset


def _prepare_dataset(path, folder, batch_size, resize_dims=None, image_size=None, map_range=None):
    def _create_map(box, kernel, image_size):
        image = np.zeros(shape=[500, 500], dtype=np.float32)

        tl = box[:2]
        br = box[2:]

        kernel_shape = np.array(np.shape(kernel)).astype(np.float32)

        center = (tl + br) / 2

        tl_in_image = center - kernel_shape / 2
        br_in_image = tl_in_image + kernel_shape

        zeros = np.array([0.0, 0.0])
        image_size = image_size[:2]

        tl_fit = np.maximum(tl_in_image, zeros)
        br_fit = np.minimum(br_in_image, image_size)

        begin_cut = tl_fit - tl_in_image
        end_cut = br_in_image - br_fit

        image_begin = center - kernel_shape / 2 + begin_cut
        image_end = center + kernel_shape / 2 - end_cut

        begin = zeros + begin_cut
        end = kernel_shape - end_cut

        image[int(image_begin[0]):int(image_end[0]), int(image_begin[1]):int(image_end[1])] = kernel[
                                                                                              int(begin[0]):int(end[0]),
                                                                                              int(begin[1]):int(end[1])]

        return image

    def _decode_image(image):
        image_string = tf.read_file(image)
        image_decoded = tf.image.decode_png(image_string, channels=3)
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

        tl_fit = tf.maximum(tl_in_image, zeros)
        br_fit = tf.minimum(br_in_image, image_shape)

        begin_cut = tl_fit - tl_in_image
        end_cut = br_in_image - br_fit

        image_begin = center - tf.cast(tf.shape(kernel), tf.float32) / 2 + begin_cut
        image_end = center + tf.cast(tf.shape(kernel), tf.float32) / 2 - end_cut

        begin = zeros + begin_cut
        end = tf.cast(tf.shape(kernel), tf.float32) - end_cut

        map = tf.Variable(tf.zeros(image_size, dtype=tf.float32))

        image_begin = tf.cast(image_begin, tf.int32)
        image_end = tf.cast(image_end, tf.int32)
        begin = tf.cast(begin, tf.int32)
        end = tf.cast(end, tf.int32)

        return tf.expand_dims(map, axis=-1)

    kernel_size = [100, 100]
    kernel = gaussian_kernel(std=20.0, size=kernel_size, norm='max')

    p = os.path.join(path, folder)

    if not os.path.exists(p):
        return None

    rgb, depth, translation, rotation, boxes = _load_data(p)

    rgb = tf.convert_to_tensor(rgb, dtype=tf.string)
    depth = tf.convert_to_tensor(depth, dtype=tf.string)
    maps = [_create_map(box, kernel, image_size) for box in boxes]

    ds = tf.data.Dataset.from_tensor_slices((rgb, depth, translation, rotation, maps))
    ds = ds.shuffle(len(rgb)).map(lambda x, y, t, r, m: [_decode_image(x), _decode_image(y, 1), t, r, m])
    ds = ds.map(lambda x, y, t, r, m: [x, y, t, r, m])

    if resize_dims is not None:
        ds = ds.map(
            lambda x, y, t, r, m: [tf.image.resize_images(x, resize_dims), tf.image.resize_images(y, resize_dims), t, r,
                                   tf.image.resize_images(tf.expand_dims(m, axis=-1), resize_dims)])

    if map_range is not None:
        ds = ds.map(lambda x, y, t, r, m: [lerp(x, *map_range), lerp(y, *map_range), t, r, m])

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
        returnBbox = list()
        for bbox in sorted(os.listdir(path)):
            returnBbox.append(os.path.join(path, bbox))
        return returnBbox

    rgb = _load_pictures(os.path.join(path, "rgb"))
    depth = _load_pictures(os.path.join(path, "depth"))
    trans, rot = _load_translations(os.path.join(path, "trans"))
    bbox = _load_bbox(os.path.join(path, 'boxes'))
    return [rgb, depth, trans, rot, bbox]


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
