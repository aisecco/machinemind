# https://www.bilibili.com/video/BV1Zt411T7zE?p=10
# https://www.bilibili.com/video/BV1Zt411T7zE?p=11&spm_id_from=pageDriver
# 续P11 10 softmax多分类
# 续P12 11 softmax多分类代码实现
# 对应P13 12 独热编码与交叉熵损失函数

# 2021-11-18 continue execute

import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

data = tf.keras.datasets.fashion_mnist.load_data()
(train_image, train_label),(test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()

print(train_image.shape)


# train_image.shape
# (60000, 28, 28)
#
# train_label.shape
# (60000,)
#
# test_image.shape, test_label.shape
# ((10000, 28, 28), (10000,))
# print(data.head())

print(test_label)
test_label_onehot = tf.keras.utils.to_categorical(test_label)
# array([[0., 0., 0., ..., 0., 0., 1.],
#        [0., 0., 1., ..., 0., 0., 0.],
#        [0., 1., 0., ..., 0., 0., 0.],
#        ...,
#        [0., 0., 0., ..., 0., 1., 0.],
#        [0., 1., 0., ..., 0., 0., 0.],
#        [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)

train_label_onehot = tf.keras.utils.to_categorical(train_label)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc']
              )
model.fit(train_image, train_label_onehot, epochs=5)
model.evaluate(test_image, test_label_onehot)

# predict
predict = model.predict(test_image)
print(predict)

for i1 in range(0, len(test_image)):
    print(i1, "label: {}, predict: {}".format(test_label[i1], np.argmax(predict[i1])))



# predict[0]
# Out[7]:
# array([8.4606875e-14, 8.5074142e-08, 2.6689962e-13, 1.2605392e-07,
#        1.6129034e-14, 1.3895242e-01, 7.3902174e-07, 1.1419640e-01,
#        1.4025960e-06, 7.4684882e-01], dtype=float32)
# np.argmax(predict[0])
# Out[8]: 9
# test_label[0]
# Out[9]: 9



