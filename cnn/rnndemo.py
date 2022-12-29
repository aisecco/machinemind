import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


#### 模拟数据生成函数：多个正弦波+随机噪音
def generate_time_series(batch_size, n_steps, seed=10):
    np.random.seed(seed)
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  # wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))  # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)  # + noise
    return series[..., np.newaxis].astype(np.float32)



#### 生成模拟时间序列以及训练集和测试集
n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, Y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, Y_test = series[9000:, :n_steps], series[9000:, -1]


def draw_xy(x, y):
    plt.figure('曲线')

    plt.plot(x, y, 'g--o', label='')

    plt.legend()  # 显示右上角的那个label,即上面的label = 'sinx'
    plt.xlabel('x')  # 设置x轴的label，pyplot模块提供了很直接的方法，内部也是调用的上面当然讲述的面向对象的方式来设置；
    plt.ylabel('y')  # 设置y轴的label;
    # plt.xlim(-1,4)                 # 可以自己设置x轴的坐标的范围哦;
    # plt.ylim(-1.5,1.2)
    plt.title('rnn data')
    plt.grid(True)  # 显示网格;

    plt.show()

print (X_train, X_train.shape)

# draw_xy(X_train[0][0], Y_train[0])
# ####朴素预测方法（naive forecast）作为预测效果比较
# Y_pred = X_valid[:, -1]
# np.mean(keras.losses.mean_squared_error(Y_valid, Y_pred))

