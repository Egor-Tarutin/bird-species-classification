import os
import argparse
import tensorflow as tf
import numpy as np

from keras.utils import load_img, img_to_array

image_shape = (224, 224)


def inference(model: tf.keras.Model, image_path: str = os.path.join('inference', '0.jpg')):
    model.trainable = False
    image = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.

    pred_probs = model.predict(img_array)

    return np.argmax(pred_probs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", help="path to load model")
    parser.add_argument("--image_name", help="name of image in inference directory")
    parser.add_argument("--image_path", help="path to image for inference")

    load_model = os.path.join('models', 'best_model.h5')
    image_path = os.path.join('inference', '0.jpg')

    args = parser.parse_args()
    if args.load_model:
        load_model = os.path.join('models', args.load_model)
    if args.image_name:
        image_path = os.path.join('inference', args.image_name)
    if args.image_path:
        image_path = str(args.image_path)

    model = tf.keras.models.load_model(load_model)
    print(inference(model, image_path))
