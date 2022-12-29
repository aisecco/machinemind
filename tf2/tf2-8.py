# https://www.bilibili.com/video/BV1Zt411T7zE?p=10
# 8 逻辑回归与交叉熵
# 9 逻辑回归实现
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
# 线性回归

import pandas as pd
#没有表头，第一行是数据
data = pd.read_csv('../credit-a.csv', header=None)

print(data.head())

x = data.iloc[:,:-1]
y = data.iloc[:,:-1].replace(-1,0)
# data.iloc[:,:-1].value_couts()

model = tf.keras.Sequential()
model.add ( tf.keras.layers.Dense(4, input_shape=(15,), activation='relu'))
model.add ( tf.keras.layers.Dense(4,  activation='relu'))
model.add ( tf.keras.layers.Dense(1,  activation='sigmoid'))
print( model.summary())
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 4)                 64
# _________________________________________________________________
# dense_1 (Dense)              (None, 4)                 20
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 5
# =================================================================
# Total params: 89
# Trainable params: 89
# Non-trainable params: 0

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc']
              )
history = model.fit(x, y, epochs=100)

history.histroy.keys()

