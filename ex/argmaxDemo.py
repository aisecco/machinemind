import tensorflow as tf
import numpy as np

#https://blog.csdn.net/u013250861/article/details/124235595?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-6-124235595-blog-117922416&spm=1001.2101.3001.4242.3&utm_relevant_index=9
# 1
print('# 1')
d1array = np.array([1,4,5,3,7,2,6])
print(np.argmax(d1array))

# 2
print('# 2')
# 1 3 5
# 0 4 3
# [0,1,0]
# [2,1]

d2array = np.array([[1,3,5],[0,4,3]])
print (d2array )
print("axis=0 ", np.argmax(d2array, axis = 0))
print("axis=1 ", np.argmax(d2array, axis = 1))



a=np.array([[2,4,5,7],[9,3,6,2]])
print('-'*30+'分割线'+'-'*30)
print(a)
print('-'*30+'分割线'+'-'*30)

a1=tf.argmax(a,axis=0)

print('tf.argmax(a,axis=0)=',a1)
print('-'*30+'分割线'+'-'*30)
a1=np.argmax(a,axis=0)
print('np.argmax(a,axis=0)=',a1)
print('-'*30+'分割线'+'-'*30)



a1=tf.argmax(a,axis=1)


print('tf.argmax(a,axis=1)=',a1)
print('-'*30+'分割线'+'-'*30)

a1=np.argmax(a,axis=1)

print('np.argmax(a,axis=1)=',a1)
print('-'*30+'分割线'+'-'*30)

