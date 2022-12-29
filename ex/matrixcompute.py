import scipy.spatial as sp
import numpy as np
import math
# In [76]: 1 - sp.distance.cdist(matrix1, matrix2, 'cosine')
# Out[76]:
# array([[ 1.        ,  0.94280904],
#        [ 0.94280904,  1.        ]])

def cos_cdist(matrix1, matrix2):
    cos = 1 - sp.distance.cdist(matrix1, matrix2, 'cosine')
    return cos

def cos_cdist_1(matrix, vector):
    v = vector.reshape(1, -1)
    return sp.distance.cdist(matrix, v, 'cosine').reshape(-1)


def cos_cdist_2(matrix1, matrix2):
    return sp.distance.cdist(matrix1, matrix2, 'cosine').reshape(-1)

def CalConDis(v1,v2,lengthVector):
     # 计算出两个向量的乘积
    B = 0
    i = 0
    while i < lengthVector:
     B = v1[i] * v2[i] + B
     i = i + 1
     # print('乘积 = ' + str(B))

     # 计算两个向量的模的乘积
     A = 0
     A1 = 0
     A2 = 0
     i = 0
     while i < lengthVector:
         A1 = A1 + v1[i] * v1[i]
         i = i + 1
     # print('A1 = ' + str(A1))

     i = 0
     while i < lengthVector:
         A2 = A2 + v2[i] * v2[i]
         i = i + 1
        # print('A2 = ' + str(A2))

     A = math.sqrt(A1) * math.sqrt(A2)
     return  format(float(B) / A,".3f")

# list1 = [[18,219,19],[17,2,5]]
# list1 = [[16,1,1],[1,2898989898,1]]
list1 = [[1,1,1],[1,2,1],[1,2,1]]
list2 = [[1,1,1],[1,2,2],[1,2,1]]


def cosSim(x, y):
    '''
    余弦相似度计算方法
    '''
    tmp = sum(a * b for a, b in zip(x, y))
    non = np.linalg.norm(x) * np.linalg.norm(y)
    return 1- tmp / float(non)


matrix1 = np.asarray(list1)
matrix2 = np.asarray(list2)

print ("cosSim:")
results0 = cosSim(matrix1, matrix2)
print (results0)

results = cos_cdist(matrix1, matrix2)
print (results)

results1 = []
for vector in matrix2:
    distance = cos_cdist_1(matrix1,vector)
    distance = np.asarray(distance)
    similarity = (1-distance).tolist()
    results1.append(similarity)
print (results1)

dist_all = cos_cdist_2(matrix1, matrix2)
results2 = []
for item in dist_all:
    distance_result = np.asarray(item)
    similarity_result = (1-distance_result).tolist()
    results2.append(similarity_result)

print (results2)

mat1 = [1,2]
mat2 = [1,2]
results3 = CalConDis(mat1, mat2, len(mat1))
print (results3)

import difflib
# a= "[instance:   *]   VM   *   (Lifecycle  Event)"
# c= "[instance:   *]   VM  Resumed   (Lifecycle  Event)"

str1= "[instance:   *]   VM  Rtopped   (Lifecycle  Event)"
str2= "[instance:   *]   VM  Resumed   (Lifecycle  Event) "
print(difflib.SequenceMatcher(None,str1,str2).ratio())
