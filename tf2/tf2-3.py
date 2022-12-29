# https://www.bilibili.com/video/BV1Zt411T7zE?p=6&spm_id_from=pageDriver
# 优化算法 梯度下降算法，学习速率
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
# 线性回归

import pandas as pd

data = pd.read_csv('../income2.csv')
print(data)

x = data.Education
y = data.Income
# plt.scatter(x, y)
plt.scatter(data.Education, data.Income)
# plt.plot(y, x)

model1 = tf.keras.Sequential()
# 输出维度为1，输入维度为1
model1.add(tf.keras.layers.Dense(1, input_shape=(1,)))
model1.summary()  # 反应整体状态

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
# adma
# mse 损失函数
model1.compile(optimizer='adam',
               loss='mse')
# 开始训练，训练循环epochs
history = model1.fit(x, y, epochs=5000)
y_predict = model1.predict(x)

# 预测x=20的输出值
x2 = pd.Series([20])
print("education={}".format(x2))

# 优化预测
#y = model1.predict(tf.data.Dataset.from_tensors(x1), batch_size=32, verbose=0)
# 不优化
y2 = model1.predict(x2)
print("predicted Income result: {}".format(y2))

# res_m = model1(x, training=False)
# res = np.array(res_m)
# print(res)

# put to end to show picture
# print(y_predict)
plt.scatter(x, y_predict)
plt.show()
