import pandas as pd
import numpy as np

# 1.1函数创建
df1 = pd.DataFrame(np.random.randn(4, 4), index=list('1234'), columns=list('ABCD'))
print(df1)

# 1.2直接创建
df2 = pd.DataFrame([[1, 2, 3],
                    [2, 3, 4],
                    [3, 4, 5]],
                   index=list('abc'), columns=list('ABC'))
print(df2)

df3 = pd.DataFrame([['20220501', 1, 2, 3],
                    ['20220502',2, 3, 4],
                    ['20220503', 3, 4, 5]],
                   index=list('123'), columns=list('ABCD'))
print(df3)

print("# 1.3 查改")
row = df3.loc[df3['A'] == '20220502']
print(row)

adate = '20220502'
df3.loc[df3['A'] == '20220502', 'B'] = 66

print(df3.loc[df3['A'] == adate].index.values.tolist()[0])

idx = df3.loc[df3['A'] == adate].index.values.tolist()[0]
df3.loc[idx , 'C'] = 88
print(df3)


# 1.3字典创建
dic1 = {
    'name': [
        '张三', '李四', '王二麻子', '小淘气'], 'age': [
        37, 30, 50, 16], 'gender': [
        '男', '男', '男', '女']}
df5 = pd.DataFrame(dic1)
print(df5)

# 2.1 查看列的数据类型
print("# 2.1 查看列的数据类型")
print(df5.dtypes)

# 2.2 查看DataFrame的头尾
df6 = pd.DataFrame(np.arange(36).reshape(6, 6), index=list('abcdef'), columns=list('ABCDEF'))
print(df6)
# 只看前2行
print("# 只看前2行")
print(df6.head(2))
# 看前5行
print("# head() 看前5行")
print(df6.head())
# 比如看后5行

print("# tail() 看后5行")
print(df6.tail())
# 比如只看后2行。
print("# tail(2) 看后2行")
print(df6.tail(2))

# 2.3 查看行名与列名
print("# .index 查看行名")
print(df6.index)
print("# .columns 查看列名")
print(df6.columns)

# 2.4 查看数据值
# 使用values可以查看DataFrame里的数据值，返回的是一个数组。
print("# 使用values可以查看DataFrame里的数据值，返回的是一个数组")
print(df6.values)

# [[ 0  1  2  3  4  5]
#  [ 6  7  8  9 10 11]
#  [12 13 14 15 16 17]
#  [18 19 20 21 22 23]
#  [24 25 26 27 28 29]
#  [30 31 32 33 34 35]]


# 查看某一列所有的数据值。
print("查看某一列所有的数据值")
print(df6['B'].values)
# [1  7 13 19 25 31]

# 2.5 查看行列数
# 使用shape查看行列数，参数为0表示查看行数，参数为1表示查看列数
print("# 2.5 使用shape查看行列数，参数为0表示查看行数，参数为1表示查看列数")
print(df6.shape[0])
print(df6.shape[1])

print("# 2.6 切片与索引")
# 使用冒号进行切片
print("df6['a':'b'")
print(df6['a':'b'])

#    A  B  C  D   E   F
# a  0  1  2  3   4   5
# b  6  7  8  9  10  11

print("# DataFrame.loc[行索引名称或条件， 列索引名称]")
# DataFrame.loc[行索引名称或条件， 列索引名称]
print(df6.loc[:, 'A':'B'])
#     A   B
# a   0   1
# b   6   7
# c  12  13
# d  18  19
# e  24  25
# f  30  31
# 切片表示的是行切片
# 索引表示的是列索引

print("# iloc 行索引和列索引的位置")
# iloc和loc的区别是，iloc接收的必须是行索引和列索引的位置，iloc方法的使用方法如下：
print(df6.iloc[0:-1, 0:2])

print("loc、iloc单列切片")
print(df6.loc[:, 'A'])  # 访问列'A'
print(df6.iloc[:, 2])  # 访问第2列

print("loc、iloc多列切片")
print(df6.loc[:, ['A', 'B']])  # A, B两列
print(df6.iloc[:, [0, 1]])  # A, B两列

print("loc、iloc花式切片")
print(df6.loc['a':'b', ['A', 'B']])  #
print(df6.iloc[1:2, [0, 1]])  #

print("loc、iloc条件切片")
print(df6.loc[df6['A'] < 22, ['A', 'B']])
print(df6.loc[df6['A'] < 22])
print(df6.loc[df6['A'] == 18])

print(df6.iloc[(df6['A'] < 22).values, [0, 1]])


# 3 DataFrame操作
print(df6.info())

# df.describe() 查看数据按列的统计信息
print(df6.describe())

# 3.3 更改DataFrame中的数据
# 更改DataFrame中的数据主要是通过将数据提取出来，重新赋值为新的数据。这种对数据的修改是无法撤销的。
#
# 首先输出CRIM列值为0.00632的数据，然后将CRIM列的值为0.00632的数据改为1，最后再查询输出CRIM列值为0.00632的数据

print(df6.loc[df6['A'] == 8].values)
df6.loc[df6['A'] == 8, 'A'] = 888
print(df6.loc[df6['A'] == 8].values)  # 因为已经修改了，所以为空值
print(df6.loc[df6['A'] == 888].values)  # 因为已经修改了，所以为空值

print(df6.loc[:, 'A'])

# 3.1 转置
print(df6.T)
#    a   b   c   d   e   f
# A  0   6  12  18  24  30
# B  1   7  13  19  25  31
# C  2   8  14  20  26  32
# D  3   9  15  21  27  33
# E  4  10  16  22  28  34
# F  5  11  17  23  29  35