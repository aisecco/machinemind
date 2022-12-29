# https://www.bilibili.com/video/BV1Zt411T7zE?p=6&spm_id_from=pageDriver
# 优化算法 梯度下降算法，学习速率
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
# 线性回归

import pandas as pd
data = pd.read_csv('../income1.csv')
print(data)

x = data.Education
y = data.Income
# plt.scatter(x, y)
plt.scatter(data.Education, data.Income)
# plt.plot(y, x)

model1 = tf.keras.Sequential()
# 输出维度为1，输入维度为1
model1.add(tf.keras.layers.Dense(1, input_shape=(1,)))
model1.summary() #反应整体状态

# print(model1)
# dense层 根据输入2个变量，输出为1
# output 第一个参数输出的维度，代表样本的个数，不需要写（None）；第二个参数
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 1)                 2
# =================================================================
# Total params: 2
# Trainable params: 2
# Non-trainable params: 0

# 优化方法
# adam
# mse 损失函数
model1.compile(optimizer='adam',
               loss='mse')
# 开始训练，训练循环epochs
history = model1.fit(x,y,epochs=2000)

y_predict = model1.predict(x)
print("y_predict:", y_predict)

# 预测x=20的输出值
x1 = 20
y1 = model1.predict( pd.Series([x1]))
print(" predicted result is: {}, (expect: {})".format(y1, x1))

plt.scatter(x, y_predict)
plt.show()




