import tensorflow as tf
import matplotlib.pyplot as plt
tf.enable_eager_execution()
import dataset.gripper_dataset as dataset
import numpy as np
path = "/home/m320/robot40human_ws/src/data_collector"
train, valid, test = dataset.get(dataset_path=path,batch_size=1, resize_dims=(320,240),image_size=(640,480))

for data in train:
    data = dataset.process(data)
    data = dataset.dictify(data)
    plt.imshow(np.squeeze(data['map']))
    plt.show()
    pass

