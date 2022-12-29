import tensorflow as tf
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

######################################
# 50题一文入门TensorFlow2.x（非Keras）之一
# https://blog.csdn.net/heywhaleshequ/article/details/104919776

##################################
# 一、Tensor张量
##################################
def normal1():
    ##############################
    # 常量
    ##############################
    # 2. 创建一个3x3的0常量张量
    c = tf.zeros([3, 3])
    print(c)

    c = tf.ones([3, 3])
    print(c)

    # 3.根据上题张量的形状，创建一个一样形状的1常量张量
    print('ones_like c')
    d = tf.ones_like(c)
    print(d)

    # 4.创建一个2x3，数值全为6的常量张量
    d = tf.fill([2, 3], 6)  # 2x3 全为 6 的常量 Tensor
    print(d)

    # 5.创建3x3随机的随机数组
    d= tf.random.normal([3, 3])
    print(d)

    # 6. 通过二维数组创建一个常量张量
    a = tf.constant([[1, 2], [3, 4]])
    print(a)

    # 7.取出张量中的numpy数组
    e = d.numpy()
    print(e)

    # 8. 从1.0 - 10.0 等间距取出5个数形成一个常量张量
    tf.linspace(1.0, 10.0, 5)

    # 9.从1开始间隔2取1个数字，到大等于10为止
    tf.range(start=1, limit=10, delta=2)

    ######################################
    # 运算
    ######################################
    # 10.将两个张量相加
    print( a + a )

    # 11.    将两个张量做矩阵乘法
    print(tf.matmul(a, a))

    # 12.两个张量做点乘
    tf.multiply(a, a)

    # 13. 将一个张量转置
    tf.linalg.matrix_transpose(c)

    # 14.将一个12x1张量变形成3行的张量
    b = tf.linspace(1.0, 10.0, 12)
    tf.reshape(b, [3, 4])

    # 方法二
    tf.reshape(b, [3, -1])

###############################
# 二、自动微分
###############################
def normal2():
    #########################################
    # 变量
    #########################################
    # 这一部分将会实现 y = x ^ 2  在 x = 1 处的导数
    # 15.新建一个1x1变量，值为1
    x = tf.Variable([1.0])  # 新建张量

    # 16.新建一个GradientTape追踪梯度，把要微分的公式写在里面
    with tf.GradientTape() as tape:  # 追踪梯度
        y = x * x
    print ( y )

    # 17.求y对于x的导数
    grad = tape.gradient(y, x)  # 计算梯度
    print ( grad)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    a = tf.constant([[3,3]])
    b = tf.constant([[2],[2]])
    c = a + b
    print(a, b, c)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(tf.__version__)

    message = tf.constant('Hello world!')
    print(message)

    normal1()
    normal2()

    print_hi('PyCharm')

