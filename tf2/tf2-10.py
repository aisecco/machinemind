# https://www.bilibili.com/video/BV1Zt411T7zE?p=10
# https://www.bilibili.com/video/BV1Zt411T7zE?p=11&spm_id_from=pageDriver
# 10 softmax多分类
# 11 softmax多分类代码实现
# 11 softmax多分类代码实现

# 2021-11-18 continue execute

import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

data = tf.keras.datasets.fashion_mnist.load_data()
(train_image, train_label),(test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()

print (train_image.shape)
# 输出一个图片
# plt.imshow(train_image[0])

# train_image.shape
# (60000, 28, 28)
#
# train_label.shape
# (60000,)
#
# test_image.shape, test_label.shape
# ((10000, 28, 28), (10000,))
# print(data.head())

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc']
              )
model.fit(train_image, train_label, epochs=10)
model.evaluate(test_image, test_label)


