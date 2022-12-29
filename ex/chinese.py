import re
import unittest

def is_chinese0(string):
    for ch in string:
        print( "is_chinese0, %s: %x" % (ch, ord(ch)))
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def is_chinese(source):
    print("source：", source)
    r = r'[\u4e00-\u9fa5]{1,}$'
    searchObj = re.search(r, source, re.M | re.I)
    if searchObj:
        print("found：")
        print("searchObj.group() : ", searchObj.group())
        total = len(searchObj.group())
        print("total：", total)
        # for i1 in range(1, total):
        #     print("searchObj.group() : ", searchObj.group(i1))
        return True
    else:
        return False

str = "17:48:30.2008710"
print(is_chinese0(str))
print(is_chinese(str))

str = "中文"
print(is_chinese(str))

str = "English"
print(is_chinese(str))

str = "中午Goodmorning"
print(is_chinese(str))
