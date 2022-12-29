# P55 输出方式
# sigmoid
# softmax 概率值的和为1
# https://www.bilibili.com/video/BV1HV411q7xD?p=55&spm_id_from=pageDriver

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

a = tf.linspace(-6, 6, 10)
si = tf.sigmoid(a)

plt.plot(a, si)
plt.grid(True)  # 显示网格;
plt.show()

a2 = tf.random.normal([1, 28, 28]) * 5
print(tf.reduce_min(a2), tf.reduce_max(a2))

a3 = tf.sigmoid(a2)
print(tf.reduce_min(a3), tf.reduce_max(a3))


a4 = tf.softmax(a2)
print(tf.reduce_min(a4), tf.reduce_max(a4))
