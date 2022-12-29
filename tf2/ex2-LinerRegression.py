import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import pickle

# 线性回归预测模型
# 本例为 tensorflow 2.x 的底层实现，不使用karas
# 生成添加随机噪声的100个 y=3x+2 直线周边的数据点，然后对这些数据点进行拟合
# https://blog.csdn.net/heywhaleshequ/article/details/104919776

# 50题一文入门TensorFlow2.x（非Keras）之二

# 损失函数，此处采用真实值与预测值的差的平方，公式为
def mean_square(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred-y_true))

# 20.创建线性回归预测模型
def linear_regression(x, W, b):
    return W * x + b

if __name__ == '__main__':
    print(tf.__version__)

    # 18.生成X,y数据，X为100个随机数，y=3X+2+noise，noise为100个随机数
    X = tf.random.normal([100, 1]).numpy()
    noise = tf.random.normal([100, 1]).numpy()

    y = 3 * X + 2 + noise

    # 19.创建需要预测的参数W, b（变量张量）
    W = tf.Variable(np.random.randn())
    b = tf.Variable(np.random.randn())

    print('W: %f, b: %f'%(W.numpy(), b.numpy()))

    # 可视化这些点
    plt.scatter(X, y)

    # 22.创建GradientTape，写入需要微分的过程
    train_steps = 20
    for i in range(train_steps):


        with tf.GradientTape() as tape:
            pred = linear_regression(X, W, b)
            loss = mean_square(pred, y)
        # 23.对loss，分别求关于W,b的偏导数
        dW, db = tape.gradient(loss, [W, b])

        # 24.用最简单朴素的梯度下降更新W, b，learning_rate设置为0.1
        W.assign_sub(0.1 * dW)
        b.assign_sub(0.1 * db)
        # print('W: %f, b: %f' % (W.numpy(), b.numpy()))
        print("step: %i, loss: %f, W: %f, b: %f" % (i + 1, loss, W.numpy(), b.numpy()))

    # 25.以上就是单次迭代的过程，现在我们要继续循环迭代20次，并且记录每次的loss,W,b
    # 画出最终拟合的曲线
    plt.plot(X, y, 'ro', label='Original data')
    plt.plot(X, np.array(W * X + b), label='Fitted line')
    plt.legend()

    # plt.plot(x_data, y_data, '*', x_data, y_pred)
    # plt.grid(True)  # 显示网格;
    plt.show()
