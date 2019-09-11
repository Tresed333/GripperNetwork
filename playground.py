import tensorflow as tf

tf.enable_eager_execution()
import dataset.gripper_dataset as dataset

from algorithms.image import coordinates
from algorithms.tensor import gaussian_kernel, vals_to_space, conv_kernel_3d, space_to_maps
from dataset.functions import *
from algorithms.norms import lerp
from algorithms.geometry import log_map
import os
import numpy as np
import matplotlib.pyplot as plt

path = "/home/m320/robot40human_ws/src/data_collector"

# rot = [np.eye(3)]
# rot = tf.Variable(rot)
# log = tf.cast(log_map(rot), tf.float32)
# space = vals_to_space(log, [1.0], min_val=-np.pi, max_val=np.pi, out_shape=[64, 64, 64])
# maps = space_to_maps(tf.expand_dims(space, axis=0))
#
# plt.imshow(maps.numpy()[0])
# plt.show()

# return {
#         'rgb': rgb,
#         'depth': depth,
#         't': trans,
#         'r': rot,
#         'params': params,
#         'map': object_map,
#         'box': box,
#         'rot_maps': rot_maps
#     }

train_dataset, val_dataset, test_dataset = dataset.get(batch_size=1, dataset_path=path, resize_dims=(320, 240),
                                                       map_range=(0.0, 255.0, 0.0, 1.0))

for step, data in enumerate(train_dataset):
    data = dataset.process(data)
    data = dataset.dictify(data)

    trans_maps = data['trans_maps'].numpy()[0]
    params = data['params'].numpy()[0]
    print(params)

    plt.imshow(trans_maps[:, :, 0])
    plt.show()
    plt.imshow(trans_maps[:, :, 1])
    plt.show()
    plt.imshow(trans_maps[:, :, 2])
    plt.show()
