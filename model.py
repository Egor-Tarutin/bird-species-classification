import tensorflow as tf
import os


def load_trainable_model(path: str = os.path.join('models', 'new_model.h5')):
    model = tf.keras.models.load_model(path)
    model.trainable = True
    for layer in model.layers:
        layer.trainable = True
    return model
