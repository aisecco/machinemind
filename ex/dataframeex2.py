import pandas as pd
import numpy as np

# 1.1函数创建
# df1 = pd.DataFrame(np.random.randn(4, 4), index=list('1234'), columns=list('ABCD'))
# print(df1)

# 1.2直接创建
# df4 = pd.DataFram([[1, 2, 3],
#                    [2, 3, 4],
#                    [3, 4, 5]],
#                   index=list('abc'), columns=list('ABC'))
# print(df4)

# 案例1 求最大前3个数
data = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6], [7, 8], [6, 8], [17, 98]]), columns=['x', 'y'], dtype=float)
Three = data.nlargest(3, 'y', keep='all')

print(Three)

print( "Max and min")
data = pd.DataFrame(np.array([[2], [34], [5], [7], [6], [981], [90]]), dtype=float)

Max0 = data.max(0)
Min0 = data.min (0)
print(Max0.item(), Max0.values, Min0.item(), Min0.values)

# 案例1 求最大前3个数
print( "Max1 and min1")
Max1 = data.nlargest(1, 0)
Min1 = data.nsmallest(1,0,keep='all')

print(Max1, Min1)