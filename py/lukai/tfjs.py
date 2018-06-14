import keras
import tensorflow as tf
from tensorflowjs.converters import keras_tfjs_loader

import lukai

def upload(config_json_path, **kwargs):
    with tf.Graph().as_default(), tf.Session() as sess:
        model = keras_tfjs_loader.load_keras_model(config_json_path)
        lukai.upload(sess, **kwargs)

