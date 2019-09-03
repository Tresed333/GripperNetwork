import tensorflow as tf
from dataset.gripper_dataset import *

tf.enable_eager_execution()

path = "/home/m320/robot40human_ws/src/data_collector"

data_train, data_valid, data_test = get(path, 1)

for data in data_train:
    data = process(data)
    data = dictify(data)
    pass
