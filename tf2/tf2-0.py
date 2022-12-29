import tensorflow as tf
from tensorflow import keras

tf.add(1, 2).numpy()

hello = tf.constant('Hello, TensorFlow!')
hello.numpy()

print(tf.__version__)
print('version: {}'.format(tf.__version__))
print(tf.keras.__version__)