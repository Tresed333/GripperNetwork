import tensorflow as tf

tf.enable_eager_execution()
from model.neural_playground.network import *
import dataset.gripper_dataset as dataset
import matplotlib.pyplot as plt


checkpoint_directory = '../../models/'
path = "/home/m320/robot40human_ws/src/data_collector"

train_dataset, val_dataset, test_dataset = dataset.get(batch_size=5, dataset_path=path, resize_dims=(320, 240),
                                                       map_range=(0.0, 255.0, 0.0, 1.0))

network = WitpNetwork(checkpoint_directory=checkpoint_directory, input_dims=(320, 240))

model = WitpModel(learning_rate=1e-4, network=network)

network.restore_model()

for data in test_dataset:
    data = dataset.process(data)
    data = dataset.dictify(data)

    outmap = model.call(inputs=data)
    out = outmap['map'][0,:,:,0]
    rgb = data["rgb"][0]
    plt.imshow(rgb.numpy())
    plt.show()

    plt.imshow(out.numpy())
    plt.show()

