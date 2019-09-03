import tensorflow as tf


def get_layers_by_names(model: tf.keras.Model, names):
    layers = list()
    for layer in model.layers:
        if layer.name in names:
            layers.append(layer)

    return layers


def vgg16(outputs):
    base_model = tf.keras.applications.vgg16.VGG16()

    layers = get_layers_by_names(base_model, outputs)
    outputs = [layer.output for layer in layers]

    model = tf.keras.Model(inputs=base_model.input,
                           outputs=outputs)
    return model


def vgg19(outputs):
    base_model = tf.keras.applications.vgg19.VGG19()

    layers = get_layers_by_names(base_model, outputs)
    outputs = [layer.output for layer in layers]

    model = tf.keras.Model(inputs=base_model.input,
                           outputs=outputs)

    return model


def resnet50(outputs):
    base_model = tf.keras.applications.resnet50.ResNet50()

    layers = get_layers_by_names(base_model, outputs)
    outputs = [layer.output for layer in layers]

    model = tf.keras.Model(inputs=base_model.input,
                           outputs=outputs)
    return model


def inception_v3(outputs):
    base_model = tf.keras.applications.inception_v3.InceptionV3()

    layers = get_layers_by_names(base_model, outputs)
    outputs = [layer.output for layer in layers]

    model = tf.keras.Model(inputs=base_model.input,
                           outputs=outputs)
    return model
