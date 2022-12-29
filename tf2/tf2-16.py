
# P16 15 Dropout抑制过拟合与网络参数选择
# P17 16 Dropout抑制过拟合

# 2021-11-26

import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

data = tf.keras.datasets.fashion_mnist.load_data()
(train_image, train_label),(test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()

# print(train_image.shape)
print(test_label)
test_label_onehot = tf.keras.utils.to_categorical(test_label)


train_label_onehot = tf.keras.utils.to_categorical(train_label)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128,activation='relu'))

model.add(tf.keras.layers.Dense(10,activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['acc']
              )
model.fit(train_image, train_label_onehot, epochs=5)
model.evaluate(test_image, test_label_onehot)

# predict
predict = model.predict(test_image)


# predict[0]
# Out[7]:
# array([8.4606875e-14, 8.5074142e-08, 2.6689962e-13, 1.2605392e-07,
#        1.6129034e-14, 1.3895242e-01, 7.3902174e-07, 1.1419640e-01,
#        1.4025960e-06, 7.4684882e-01], dtype=float32)
# np.argmax(predict[0])
# Out[8]: 9
# test_label[0]
# Out[9]: 9



