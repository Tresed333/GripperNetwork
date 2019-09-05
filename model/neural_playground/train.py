import tensorflow as tf

tf.enable_eager_execution()
from tensorflow.contrib import summary as summary
from model.summaries import BaseLogs
import dataset.gripper_dataset as dataset
import os
from model.neural_playground.network import *
from algorithms.geometry import *
from algorithms.progress_bar import *

def compare_translation(inputs, outputs):
    in_params = inputs["params"]
    out_params = outputs["params"]
    in_values = in_params[:, :3]
    out_values = out_params[:, :3]

    diff = tf.cast(in_values, tf.float32) - out_values
    sqr = diff * diff
    sums = tf.reduce_sum(sqr, axis=1)
    sqrt = tf.sqrt(sums)
    return tf.reduce_mean(sqrt)

def compare_rotation(inputs, outputs):
    in_params = inputs["params"]
    out_params = outputs["params"]
    in_values = tf.cast(in_params[:, 3:], tf.float32)
    out_values = out_params[:, 3:]

    in_rot = exp_map(in_values)
    out_rot = exp_map(out_values)

    angle = rotation_delta(in_rot, out_rot)
    return tf.reduce_mean(angle)*180.0/np.pi

class Logs(BaseLogs):

    def summary(self, inputs, outputs, losses, step):
        super().summary(inputs, outputs, losses, step)

        with summary.always_record_summaries():
            if step.numpy() % 20 == 0:
                summary.scalar('summary/loss', losses['loss'], step=step)
                summary.scalar('summary/compare_trans', compare_translation(inputs, outputs), step=step)
                summary.scalar('summary/compare_rot', compare_rotation(inputs, outputs), step=step)


checkpoint_directory = '../../models/'
log_directory = '../../logs/GripperNetwork/'

epochs = 200
lr = 1e-4
path = "/home/m320/robot40human_ws/src/data_collector"

train_step = tf.Variable(0, dtype=tf.int64, trainable=False)
val_step = tf.Variable(0, dtype=tf.int64, trainable=False)
global_step = tf.train.get_or_create_global_step()

train_dataset, val_dataset, test_dataset = dataset.get(batch_size=5, dataset_path=path, resize_dims=(320,240), map_range=(0.0,255.0,0.0,1.0))

network = GripperNetwork(checkpoint_directory=checkpoint_directory, input_dims=(320, 240))

model = GripperModel(learning_rate=lr, network=network)

train_logs = Logs(os.path.join(log_directory, 'train'))
val_logs = Logs(os.path.join(log_directory, 'val'))

network.restore_model()

for e in range(epochs):
    print('Epoch: ', e)
    for step, data in enumerate(train_dataset):
        data = dataset.process(data)
        data=dataset.dictify(data)
        inputs = data

        with tf.GradientTape(persistent=True) as tape:
            outputs = model(inputs, training=True)
            losses = model.compute_loss(inputs, outputs)
        model.optimize(losses, tape)

        train_logs.summary(inputs, outputs, losses, train_step)
        train_step = train_step + 1
    model.save_model(e)

    if test_dataset is not None:
        for step, data in enumerate(test_dataset):
            data = dataset.process(data)
            data = dataset.dictify(data)
            inputs = data
            outputs = model(inputs, training=False)
            losses = model.compute_loss(inputs, outputs)

            val_logs.summary(inputs, outputs, losses, val_step)
            val_step = val_step + 1
