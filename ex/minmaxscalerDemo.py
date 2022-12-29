import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = np.array([[4, 2, 3],
                 [1, 5, 6]])

# 手动归一化
feature_range = (0, 1)  # 要映射的区间
print(data.min(axis=0))
print(data.max(axis=0))
x_std = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
x_scaled = x_std * (feature_range[1] - feature_range[0]) + feature_range[0]
print('手动归一化结果：\n{}'.format(x_scaled))

# 自动归一化
scaler = MinMaxScaler()
print('自动归一化结果:\n{}'.format(scaler.fit_transform(data)))

# 归一化
sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化：归一化到(0，1)之间

training_set = np.array([[40.1, 20.8, 30.8],
                         [1.56, 50.1, 6.1]])
test_set = np.array([[4.2, 2.1, 3.0],
                     [1.0, 5.0, 6.0]])
training_set_scaled = sc.fit_transform(training_set)

print('\ntrainning 归一化结果:\n{}'.format(training_set_scaled))
test_set_scaled = sc.transform(test_set)  # 利用训练集的属性对测试集进行归一化
print('test 归一化结果:\n{}'.format(test_set_scaled))

predicted_stock_price = sc.inverse_transform(training_set_scaled)
print('trainning 反归一化结果:\n{}'.format(predicted_stock_price))

# 对真实数据还原---从（0，1）反归一化到原始范围
real_stock_price = sc.inverse_transform(test_set_scaled)
print('test 反归一化结果:\n{}'.format(real_stock_price))

# 一维，其实还是2维
print("\n####一维，其实还是2维:")
training_set = np.array([[50.1], [1.56], [1.56], [555.0], [10000.0]])
test_set = np.array([[8.2], [8.2], [50.1], [1668.2], [123888.0]])
training_set_scaled = sc.fit_transform(training_set)

print('\ntrainning 归一化结果:\n{}'.format(training_set_scaled))
test_set_scaled = sc.transform(test_set)  # 利用训练集的属性对测试集进行归一化
print('test 归一化结果:\n{}'.format(test_set_scaled))

predicted_stock_price = sc.inverse_transform(training_set_scaled)
print('trainning 反归一化结果:\n{}'.format(predicted_stock_price))

# 对真实数据还原---从（0，1）反归一化到原始范围
real_stock_price = sc.inverse_transform(test_set_scaled)
print('test 反归一化结果:\n{}'.format(real_stock_price))
