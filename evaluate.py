import os
import argparse
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

image_shape = (224, 224)


def evaluate(model: tf.keras.Model, data_path: str = 'data', batch_size: int = 32):
    model.trainable = False

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_path = os.path.join(data_path, 'test')

    test_generator = test_datagen.flow_from_directory(test_path, target_size=image_shape, batch_size=batch_size,
                                                      class_mode='categorical')
    results = model.evaluate(test_generator)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="path to dataset")
    parser.add_argument("--batch_size", help="batch size")
    parser.add_argument("--load_model", help="path to load model")

    data_path = 'data'
    batch_size = 32
    load_model = os.path.join('models', 'best_model.h5')

    args = parser.parse_args()
    if args.data:
        data_path = str(args.data)
    if args.batch_size:
        batch_size = int(args.batch_size)
    if args.load_model:
        load_model = str(args.load_model)

    model = tf.keras.models.load_model(load_model)
    print(evaluate(model=model, data_path=data_path, batch_size=batch_size))
