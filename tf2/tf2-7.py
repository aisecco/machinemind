# https://www.bilibili.com/video/BV1Zt411T7zE?p=6&spm_id_from=pageDriver
# 6. 多层感知器（神经网络）与激活函数
# 7. 多层感知器（神经网络）的代码实现
# 8. 逻辑回归实现
# 通过对广告数据和销量数据分析进行多感知器的实现
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv('../ads.csv')
print(data.head())

x = data.TV
y = data.sales
plt.scatter(x, y)
plt.show()

plt.scatter(data.radio, y)
plt.show()

plt.scatter(data.newspaper, y)
plt.show()


# 中间三列
x = data.iloc[:,1:-1]
# 最后一列
y = data.iloc[:, -1]

# 3个维度输入
# ax1 + bx2 + cx3 + d
# dense 隐含层数量（第一个参数）
# input_shape 3 为 三种数据输入

# activation激活层,中间层激活，增加非线性关系，提高拟合能力
model1 = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(3,), activation='relu'),
                             tf.keras.layers.Dense(1)]
                             )
model1.summary() #反应整体状态


# 优化方法
# adma
# mse 损失函数
model1.compile(optimizer='adam',
               loss='mse')
# 开始训练，训练循环epochs
history = model1.fit(x,y,epochs=200)

test = data.iloc[:10, 1:-1]
y_predict = model1.predict(test)

print(y_predict)
y = data.iloc[:10, -1]


plt.scatter(y_predict , y)







