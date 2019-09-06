import tensorflow as tf

tf.enable_eager_execution()
from tensorflow.contrib import summary as summary
from model.summaries import BaseLogs
import dataset.gripper_dataset as dataset
import matplotlib.pyplot as plt
import os
from model.neural_playground.network import WitpNetwork
import numpy as np

checkpoint_directory = '../../models/'
path = "/home/rafal/Datasets/tube"
train_dataset, val_dataset, test_dataset = dataset.get(dataset_path=path, batch_size=5,
                                                       map_range=[0.0, 255.0, 0.0, 1.0])

# network = RotateNet(checkpoint_directory=checkpoint_directory, suffix='bottle')

# model = MnistClassifier(checkpoint_directory=checkpoint_directory)

# model.restore_model()

for data in test_dataset:
    data = dataset.process(data)
    data = dataset.dictify(data)
    # print(data['box'][0])
    # rgb = data["rgb"][0]
    # map = data["map"][0, :, :, 0]
    # plt.imshow(rgb.numpy())
    # plt.show()
    #
    # plt.imshow(map.numpy())
    # plt.show()
    pass
