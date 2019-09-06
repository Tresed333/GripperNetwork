import os
from tensorflow.contrib import eager as tfe
from system.misc import makedirs
from algorithms.image import *


class GripperNetwork(tf.keras.Model):
    MODEL_NAME = 'GripperNetwork'

    def __init__(self, input_dims=(28, 28), checkpoint_directory=None, suffix='', learning_rate=1e-4):
        super(GripperNetwork, self).__init__()

        self.checkpoint_directory = checkpoint_directory
        self.suffix = suffix

        self.input_dims = input_dims

        self.c1 = tf.keras.layers.Conv2D(16, [3, 3], activation='relu', padding='same')
        self.c2 = tf.keras.layers.Conv2D(32, [3, 3], activation='relu', padding='same')
        self.c3 = tf.keras.layers.Conv2D(32, [3, 3], activation='relu', padding='same')
        self.c4 = tf.keras.layers.Conv2D(64, [3, 3], activation='relu', padding='same')
        self.c5 = tf.keras.layers.Conv2D(32, [3, 3], activation='relu', padding='same')
        self.c6 = tf.keras.layers.Conv2D(16, [3, 3], activation='relu', padding='same')
        self.dense_conv = tf.keras.layers.Dense(128)
        self.dense_out = tf.keras.layers.Dense(6)
        self.max_pool = tf.keras.layers.MaxPool2D()
        self.flatten = tf.keras.layers.Flatten()

    def call(self, rgb, depth, mesh, training=None, mask=None):
        x = tf.concat((rgb, depth, mesh), axis=-1)
        x = self.c1(x)
        x = self.c2(x)
        x = self.max_pool(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.max_pool(x)
        x = self.c5(x)
        x = self.c6(x)
        x = self.flatten(x)
        x = self.dense_conv(x)
        x = self.dense_out(x)
        return x

    def restore_model(self):
        """ Function to restore trained model.
        """

        self(tf.zeros((1,) + self.input_dims + (3,)), tf.zeros((1,) + self.input_dims + (1,)), training=False)
        try:
            saver = tfe.Saver(self.variables)
            saver.restore(
                tf.train.latest_checkpoint(
                    os.path.join(self.checkpoint_directory, GripperNetwork.MODEL_NAME + self.suffix)))
        except ValueError:
            print('RotateNet model cannot be found.')

    def save_model(self, step):
        """ Function to save trained model.
        """
        makedirs(os.path.join(self.checkpoint_directory, GripperNetwork.MODEL_NAME))
        tfe.Saver(
            self.variables).save(
            os.path.join(self.checkpoint_directory, GripperNetwork.MODEL_NAME + self.suffix,
                         GripperNetwork.MODEL_NAME),
            global_step=step)

class GripperModel(tf.keras.Model):

    def __init__(self, gripperNetwork, witpNetwork, learning_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gripperNetwork = gripperNetwork
        self.witpNetwork = witpNetwork
        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

    def call(self, inputs, training=None, mask=None):
        rgb = inputs['rgb']
        depth = inputs['depth']
        witpout = self.witpNetwork(rgb, depth, training=training)
        gripperout = self.gripperNetwork(rgb, depth, witpout, training=training)

        return {
            'params': gripperout
        }

    def compute_loss(self, inputs, outputs):
        pred = outputs['params']
        label = inputs["params"]
        loss = tf.losses.mean_squared_error(label, pred)

        return {
            'loss': loss,
        }

    def optimize(self, losses, tape, global_step=None):
        def _apply_grads(loss, tape, var, opt, step):
            grads = tape.gradient(loss, var)
            opt.apply_gradients(zip(grads, var), global_step=step)

        _apply_grads(losses['loss'], tape,
                     self.trainable_variables, self.opt, global_step)

    def save_model(self, step):
        """ Function to save trained model.
        """
        self.witpNetwork.save_model(step)




class WitpNetwork(tf.keras.Model):
    MODEL_NAME = 'WitpNetwork'

    def __init__(self, input_dims=(28, 28), checkpoint_directory=None, suffix='', learning_rate=1e-4):
        super(WitpNetwork, self).__init__()

        self.checkpoint_directory = checkpoint_directory
        self.suffix = suffix

        self.input_dims = input_dims

        self.e1 = tf.keras.layers.Conv2D(16, [3, 3], activation='relu', padding='same')
        self.e2 = tf.keras.layers.Conv2D(32, [3, 3], activation='relu', padding='same')
        self.e3 = tf.keras.layers.Conv2D(32, [3, 3], activation='relu', padding='same')
        self.e4 = tf.keras.layers.Conv2D(64, [3, 3], activation='relu', padding='same')
        self.e5 = tf.keras.layers.Conv2D(32, [3, 3], activation='relu', padding='same')
        self.e6 = tf.keras.layers.Conv2D(16, [3, 3], activation='relu', padding='same')
        self.e7 = tf.keras.layers.Conv2D(16, [3, 3], activation='relu', padding='same')
        self.e8 = tf.keras.layers.Conv2D(16, [3, 3], activation='relu', padding='same')
        self.e9 = tf.keras.layers.Conv2D(16, [3, 3], activation='relu', padding='same')
        self.e10= tf.keras.layers.Conv2D(16, [3, 3], activation='relu', padding='same')

        self.max_pool = tf.keras.layers.MaxPool2D()

        self.d1 = tf.keras.layers.Conv2DTranspose(64, [3, 3], activation='relu', padding='same',strides=2)
        self.d2 = tf.keras.layers.Conv2DTranspose(32, [3, 3], activation='relu', padding='same',strides=2)
        self.d3 = tf.keras.layers.Conv2DTranspose(32, [3, 3], activation='relu', padding='same',strides=2)
        self.d4 = tf.keras.layers.Conv2DTranspose(16, [3, 3], activation='relu', padding='same',strides=2)

        self.fin= tf.keras.layers.Conv2D(1, [3, 3], activation='relu', padding='same')

    def call(self, rgb, depth, training=None, mask=None):
        x = tf.concat((rgb, depth), axis=-1)
        x = self.e1(x)
        x = self.e2(x)
        x = self.max_pool(x)
        x = self.e3(x)
        x = self.e4(x)
        x = self.max_pool(x)
        x = self.e5(x)
        x = self.e6(x)
        x = self.max_pool(x)
        x = self.e7(x)
        x = self.e8(x)
        x = self.max_pool(x)
        x = self.e9(x)
        x = self.e10(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.fin(x)
        return x

    def restore_model(self):
        """ Function to restore trained model.
        """

        self(tf.zeros((1,) + self.input_dims + (3,)), tf.zeros((1,) + self.input_dims + (1,)), training=False)
        try:
            saver = tfe.Saver(self.variables)
            saver.restore(
                tf.train.latest_checkpoint(
                    os.path.join(self.checkpoint_directory, WitpNetwork.MODEL_NAME + self.suffix)))
        except ValueError:
            print('RotateNet model cannot be found.')

    def save_model(self, step):
        """ Function to save trained model.
        """
        makedirs(os.path.join(self.checkpoint_directory, WitpNetwork.MODEL_NAME))
        tfe.Saver(
            self.variables).save(
            os.path.join(self.checkpoint_directory, WitpNetwork.MODEL_NAME + self.suffix,
                         WitpNetwork.MODEL_NAME),
            global_step=step)


class WitpModel(tf.keras.Model):
    def __init__(self, network, learning_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = network
        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

    def call(self, inputs, training=None, mask=None):
        rgb = inputs['rgb']
        depth = inputs['depth']
        out = self.network(rgb, depth, training=training)

        return {
            'map': out
        }

    def compute_loss(self, inputs, outputs):
        pred = outputs['map']
        label = inputs['map']
        loss = tf.losses.mean_squared_error(label, pred)

        return {
            'loss': loss,
        }

    def optimize(self, losses, tape, global_step=None):
        def _apply_grads(loss, tape, var, opt, step):
            grads = tape.gradient(loss, var)
            opt.apply_gradients(zip(grads, var), global_step=step)

        _apply_grads(losses['loss'], tape,
                     self.trainable_variables, self.opt, global_step)

    def save_model(self, step):
        """ Function to save trained model.
        """
        self.network.save_model(step)