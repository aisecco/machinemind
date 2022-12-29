import numpy as np

arr = [1,2,3,4,5,6]

# mean
arr_mean = np.mean(arr)
print (arr_mean)

arr_var = np.var(arr)
print (arr_var)

arr_std_1 = np.std(arr)
print (arr_std_1)

arr_std_2 = np.std(arr, ddof=1)
print (arr_std_2)
