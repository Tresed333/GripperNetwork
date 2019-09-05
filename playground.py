import tensorflow as tf


from algorithms.geometry import *
from algorithms.transforms import euler_to_rot
import numpy as np

# rot_1 = np.eye(3)
# rot_2 = euler_to_rot(xyz=np.deg2rad([90.0, 0, 0]))
#
# rot_1 = np.reshape(rot_1, (1, 3, 3))
# rot_2 = np.reshape(rot_2, (1, 3, 3))
#
# log1 = log_map(rot_1)
# log2 = log_map(rot_2)
#
# print(log1)
# print(log2)

# input = {"params": tf.constant([
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#
# ])
# }
#
# output = {"params": tf.constant([
#     [0.023, 0.054, 0.0532, np.pi/2, 0, 0],
#     [0.022, 0.044, 0.055, np.pi/2, 0, 0],
#     [0.023, 0.054, 0.0532, np.pi/2, 0, 0]
#
# ])
# }
#
#
# def compare_translation(inputs, outputs):
#     in_params = inputs["params"]
#     out_params = outputs["params"]
#     in_values = in_params[:, :3]
#     out_values = out_params[:, :3]
#
#     diff = in_values - out_values
#     sqr = diff * diff
#     sums = tf.reduce_sum(sqr, axis=1)
#     sqrt = tf.sqrt(sums)
#     return tf.reduce_mean(sqrt)
#
#
# def compare_rotation(inputs, outputs):
#     in_params = inputs["params"]
#     out_params = outputs["params"]
#     in_values = in_params[:, 3:]
#     out_values = out_params[:, 3:]
#
#     in_rot = exp_map(in_values)
#     out_rot = exp_map(out_values)
#
#     angle = rotation_delta(in_rot, out_rot)
#     return tf.reduce_mean(angle)
#
#
# if __name__ == '__main__':
#     a = compare_rotation(input, output)
#     print(a)
import os
def _load_bbox(path):
    returnBbox = list()
    for bbox in sorted(os.listdir(path)):
        returnBbox.append(os.path.join(path, bbox))
    return returnBbox

if __name__ == '__main__':
    path = "/home/m320/robot40human_ws/src/data_collector/rgb_bbox"
    list = _load_bbox(path)
    print(list)