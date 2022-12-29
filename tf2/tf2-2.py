import numpy as np
import matplotlib.pyplot as plt

# test read npz file
a = np.load('/Users/mac/dev/python/tf/mnist.npz')
print(a.files)

x_train = a['x_train']
y_train = a['y_train']
x_test = a['x_test']
y_test = a['y_test']

# uint8
print(x_train.dtype)
print(x_train[0])

x_data = np.linspace(0, 10, 20)  + np.random.uniform(-1.5,1.5,20)
y_data = np.linspace(0, 10, 20)  + np.random.uniform(-1.5,1.5,20)

# x_data = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]
# y_data = [0, 0.5, 1.2, 1.7, 1.8, 2.5, 3.1, 3.5, 4.0, 4.5, 5.0, 5.5, 6.1, 6.5, 7.1, 7.5, 8.1, 8.5, 9.0]

# plt.plot(x_data, y_data, '*', x_data, y_pred)
plt.plot(x_data, y_data, '*')


x = np.arange(1, 11)
y = 2 * x + 5
plt.title("Matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
# plt.plot(x, y)
# plt.show()

plt.grid(True)  # 显示网格;
plt.show()