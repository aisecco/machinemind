from scipy.spatial import distance
import numpy as np
import time

a_float32 = np.empty((1000, 512), dtype=np.float32)
b_float32 = np.empty((1, 512), dtype=np.float32)
a_float64 = np.empty((1000, 512), dtype=np.float64)
b_float64 = np.empty((1, 512), dtype=np.float64)

test_total = 1

t1 = time.time()
for i in range(test_total):
    distance.cdist(a_float32, b_float32, "sqeuclidean")
t2 = time.time()
print("float32 mode:", t2 - t1)


t1 = time.time()
for i in range(test_total):
    distance.cdist(a_float64, b_float64, 'sqeuclidean')
t2 = time.time()
print("float64 mode:", t2 - t1)
