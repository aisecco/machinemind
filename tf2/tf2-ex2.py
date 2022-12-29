import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = tf.random.normal([100, 1]).numpy()
noise = 0 # tf.random.normal([100, 1]).numpy()

y = 3*X+2+noise

plt.scatter(X, y)
plt.show()

# 自己建数据集
x = tf.compat.v1.random_normal([100,1])*50
y = tf.compat.v1.random_normal([100,1])*10
z = tf.compat.v1.random_normal([100,1])
f = 0.5*x+5*y-2*z
