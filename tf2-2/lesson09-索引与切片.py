# P25-27

import tensorflow as tf

tf.random.normal([4,28,28,3])
a=tf.random.normal([4,28,28,3])

# Out[5]: TensorShape([4, 28, 28, 3])
print(a[0].shape)
# Out[6]: TensorShape([28, 28, 3])
print(a[0,2].shape)

print(a[:,:,:,1], a[:,:,:,2].shape)
# Out[7]: TensorShape([28, 3])
print(a[0,2,3].shape)
# Out[8]: TensorShape([3])
print(a[1,2,3].shape)
# Out[9]: TensorShape([3])
print(a[1,2,3,1].shape)
# Out[10]: TensorShape([])


#P28 tf.gather
# 随意选取样本
a=tf.range(10)
tf.gather(a, axis=0, indices=[2,3]).shape



















































