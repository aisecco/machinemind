import numpy as np
from matplotlib import pyplot as plt
# tsv

import pandas as pd

# data = pd.read_csv('income2.csv')
data = pd.read_csv("../data/201802171302.txt", encoding="utf-8", header=None, sep="|",dtype={},keep_date_col=True)
# df = pd.read_csv('data.csv', sep='\t',header=None, names=['var_code','var_name','var_desc'])


# print(data)
print(data.index)
rows = data.shape[0]
cols = data.columns.shape[0]

print(data.dtypes)

for i in range(0, 10):
    for j in range(0, cols):
        # print(j,type(data.iloc[i][j]), data.iloc[i][j])
        # if type(data.iloc[i][j]) == str and len(data.iloc[i][j]) > 0:
        #     print(j, data.iloc[i][j])
        if type(data.iloc[i][j]) == float and data.iloc[i][j] :
            print(j, data.iloc[i][j])

# row_data_3 = data.iloc[0:2]
# print(row_data_3)
#
# row_data_5 = data.iloc[[0,2]]
# print(row_data_5)

