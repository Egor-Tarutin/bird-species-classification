import os
import argparse
import tensorflow as tf
import pickle
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.metrics import Precision, Recall
from model import load_trainable_model

image_shape = (224, 224)
learning_rate = 1e-3


def scheduler(epoch, lr):
    return lr if epoch < 40 else lr * tf.math.exp(-0.1)


def train(model: tf.keras.Model,
          data_path: str = 'data',
          batch_size: int = 32,
          num_epochs: int = 50,
          save_model: str = os.path.join('models', 'saved_model.h5'),
          save_logs: str = os.path.join('logs', 'saved_logs.pkl')
          ) -> None:
    model.trainable = True
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-4:]:
        layer.trainable = True

    train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, brightness_range=[0.4, 1.2])
    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')

    train_generator = train_datagen.flow_from_directory(train_path, target_size=image_shape, batch_size=batch_size,
                                                        class_mode='categorical')
    validation_generator = valid_datagen.flow_from_directory(val_path, target_size=image_shape, batch_size=batch_size,
                                                             class_mode='categorical')

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy',
                  metrics=['accuracy', Precision(), Recall()])

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    history = model.fit(train_generator, validation_data=validation_generator, epochs=num_epochs,
                        callbacks=[callback])

    with open(save_logs, 'wb') as f:
        pickle.dump(history.history, f)

    model.save(save_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="path to dataset")
    parser.add_argument("--batch_size", help="batch size")
    parser.add_argument("--num_epochs", help="number of epochs")
    parser.add_argument("--save_model", help="path where model will be saved")
    parser.add_argument("--save_logs", help="path where logs will be saved")

    data_path = 'data'
    save_model = os.path.join('models', 'saved_model.h5')
    save_logs = os.path.join('logs', 'saved_logs.pkl')
    batch_size = 32
    num_epochs = 50

    args = parser.parse_args()
    if args.data:
        data_path = str(args.data)
    if args.batch_size:
        batch_size = int(args.batch_size)
    if args.num_epochs:
        num_epochs = int(args.num_epochs)
    if args.save_model:
        save_model = os.path.join('models', args.save_model)
    if args.save_logs:
        save_logs = os.path.join('logs', args.save_logs)

    train(model=load_trainable_model(),
          data_path=data_path,
          batch_size=batch_size,
          num_epochs=num_epochs,
          save_model=save_model,
          save_logs=save_logs)
