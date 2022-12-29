import os

srcFile = '../data/test1-copy.txt'
dstFile = '../data/test1.txt'
try:
    os.rename(srcFile,dstFile)
except Exception as e:
    print(e)
    print('rename file failed\r\n')
else:
    print('rename file success\r\n')
