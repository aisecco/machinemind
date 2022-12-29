
# P18 17 函数式API与多输入输出模型 1
# P19 18 函数式API与多输入输出模型 2
# 2021-11-27

# 这个示例开始为日月光华的视频教程中公开部分，但是没有讲完，只讲到了lamda表达式。后来我从网上找到了文章版，按照文章版完整做下来 2021-12-06
# https://blog.csdn.net/cfan927/article/details/103438591?spm=1001.2101.3001.6650.11&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-11.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-11.no_search_link

import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# Fashion MNIST数据集中包含10个类别的70,000张衣物的灰度图像，分辨率为28×28像素
# Fashion MNIST数据集的目的是替代经典的MNIST数据集（这类似于编程学习里面的“Hello World”。经典的MNIST数据集包含手写数字（0、1、2等）的图像，其格式与这里使用的衣物数据集相同。
# 这里，使用60,000张图像来训练网络，使用10,000张图像来评估网络图像分类网络的准确性。通过下列语句可以直接从TensorFlow访问、导入和加载Fashion MNIST数据：
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()

# 上述代码输出：
# Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
# 32768/29515 [=================================] - 0s 0us/step
# Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
# 26427392/26421880 [==============================] - 0s 0us/step
# Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
# 8192/5148 [===============================================] - 0s 0us/step
# Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
# 4423680/4422102 [==============================] - 0s 0us/step

# 上述语句从TensorFlow加载数据集并返回四个NumPy数组：
#
# train_images和train_labels数组是训练集：是模型用来学习的数据；
# test_images和test_labels数组是测试集：通过测试集对模型进行测试。

# 这些图像是28x28维的NumPy数组，像素值分布在0到255之间。标签（Label）是一个整数数组，范围从0到9，与图像所代表的衣物类别相对应：
# Label	Class
# 0	T-shirt/top
# 1	Trouser
# 2	Pullover
# 3	Dress
# 4	Coat
# 5	Sandal
# 6	Shirt
# 7	Sneaker
# 8	Bag
# 9	Ankle boot

# 每幅图像都对应一个标签。由于类名没有包含在数据集中，所以将它们存储在class_names中，以便后面绘制图像时使用:
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 在训练模型之前，让我们先研究一下数据集的格式。可以通过以下命令输出数据集的部分细节：
# train_images.shape
# len(train_labels)
# train_labels
# test_images.shape
# len(test_labels)
#
# 对应输出如下：
# (60000, 28, 28)                                  - 训练集中有60000张图像，每张图像的大小都为28x28像素
# 60000                                            - 训练集中有60000个标签
# array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)      - 每个标签都是0到9之间的整数
# (10000, 28, 28)                                  - 测试集中有10,000张图像。同样，每个图像表示为28 x 28像素
# 10000                                            - 测试集包含10,000个图像标签

# 3 处理数据
# 在训练网络之前，必须对数据进行预处理。如果你检查训练集中的第一个图像，你会看到像素值落在0到255的范围内。可通过下列代码显示图像：

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# 为了验证数据的格式是否正确，我们通过以下代码显示来自训练集的前25个图像，并在每个图像下面显示类名：

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# 在将这些值输入神经网络模型之前，需要将训练集和测试集中图像的像素值缩放到0到1的范围
train_images = train_images / 255.0
test_images = test_images / 255.0

# 4 构建（配置）模型
# 建立神经网络需要配置模型的层，然后编译模型，下面分别实现。
#
# 4.1 设置模型的层（layer）
# layer是神经网络的基本组件，它从输入的数据中提取数据的特征表示。

# 大多数深度学习是由简单的layer链接在一起构成的。大多数layer（如tf.keras.layers.Dense），包含有在训练中学习的参数。
# 我们使用以下代码构建本节的模型：
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# tf.keras.layers.Flatten：这是网络的第一层，将图像的格式从一个二维数组（28×28像素）转换为一个一维数组（28 * 28 = 784像素）。可以把这个层看作是将图像中的像素行分解并排列起来。这一层没有需要学习的参数，它只是重新格式化数据。
# 当像素被格式化后，网络由两个tf.keras.layers.Dense组成的序列组成。这些layer紧密相连，或者说完全相连：
# # 第一个Dense层有128个节点（或神经元）；
# # 第二个Dense层（也即最后一层）是一个有10个节点的softmax层，它返回一个10个概率值的数组，这些概率值的和为1。每个节点包含一个分数，表示当前图像属于10个类之一的概率。

# input = tf.keras.Input(shape=(28,28))
# x = tf.keras.layers.Flatten()(input)
#
# # 多输入, 2个input 1个output
# # input2 = tf.keras.Input(shape=(28,28))
# # x2 = tf.keras.layers.Flatten()(input2)
# # x = tf.keras.concatenate([x, x2 ])
#
# x = tf.keras.layers.Dense(32, activation='relu')(x)
# x = tf.keras.layers.Dropout(0.5)(x)
# x = tf.keras.layers.Dense(64, activation='relu')(x)
# output = tf.keras.layers.Dense(10, activation='softmax')(x)
#
# model = tf.keras.Model(input)

# 4.2 编译模型
# 在对模型进行训练之前，需要额外设置一些参数。这些是在模型的编译步骤中添加的：
#
# 损失函数（Loss function）：用来衡量训练过程中模型的准确性，模型训练时通过最小化这个函数来”引导“模型朝正确的方向前进；
# 优化器（Optimizer）：是模型根据数据和损失函数进行更新的方式；
# 度量（Metrics）：用于监视训练和测试步骤。下面的例子使用accuracy度量，即被正确分类的图像的比例。
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5 训练模型
# 训练神经网络模型需要以下步骤：
#
# 将训练数据输入模型。在本例中，训练数据存放在train_images和train_tags数组中；
# 模型通过学习把图像和标签联系起来；
# 让模型对本例中的测试集test_images数组进行预测。验证预测是否与test_labels数组中的标签匹配。
# 要开始训练，使用model.fit方法：

model.fit(train_images, train_labels, epochs=50)
print( "train_labels[0]:", train_labels[0] )

# 当模型训练时，会输出显示损失（loss）和精度（accuracy）度量指标。该模型的精度约为0.91（或91%）。
# 6 评估模型精度
# 接下来，比较模型在测试数据集上的执行情况：

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# 10000/1 - 1s - loss: 0.2934 - accuracy: 0.8830
# Test accuracy: 0.883
# 结果表明，测试数据集的准确性略低于训练数据集的准确性。这种训练精度和测试精度之间的差距表示过拟合。过拟合是指机器学习模型在新的、以前未见过的输入上的表现不如在训练数据上的表现。

# 7 模型预测
# 通过训练模型，可以使用它对一些图像进行预测：
#
predictions = model.predict(test_images)
# 1
# 这里，模型已经预测了测试集中每张图片的标签，让我们看一下第一个预测：

# predictions[0]
# 输出如下：
#
# array([1.06123218e-06, 8.76374884e-09, 4.13958730e-07, 9.93547733e-09,
#    2.39135318e-07, 2.61428091e-03, 2.91701099e-07, 6.94991834e-03,
#    1.02351805e-07, 9.90433693e-01], dtype=float32)

# 预测结果是一个由10个数字组成的数组。它们代表了模特的“置信度（confidence）”，即图像对应于10件不同的衣服中的每一件。你可以看到哪个标签的置信度最高：
# np.argmax(predictions[0])
# 输出为：
# 9
# 因此，模型最确信这是一个Ankle boot，或class_names[9]。


# 我们将这张图绘制出来查看完整的10个类预测的置信度，下面定义2个函数用于绘制数据图像：
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# 让我们看看第0张图片、预测和预测数组。正确的预测标签是蓝色的，错误的预测标签是红色的。这个数字给出了预测标签的百分比（满分100）。调用前面定义的函数来绘制数据图：
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()