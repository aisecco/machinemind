import time
import re
import sys
import unittest


DEBUG = False
# 0 string, is the origin string. preserve this value.
# 1-9 is the base type, > 10 is higher priority type
# type no, regular expression pattern, priority

patterns = [
    [1, r'^-?\d+$', 'number'],  # num
    [2, r'^(-?\d+)(\.\d+)?$', 'float'],  # float
    [3, r'^-?([1-9]\\d*\\.\\d*|0\\.\\d*[1-9]\\d*|0?\\.0+|0)$', 'double'],  # double
    [4, r'^20\d{2}-?[01]\d-?[0-3]\d\s?[012][0-9]:?[0-5][0-9]:?[0-5][0-9].?', 'datetime'],  # date
    [4, r'^20\d{2}-?[01]\d-?[0-3]\d\s?([012][0-9]:?[0-5][0-9]:?[0-5][0-9].?)*', 'datetime'],  # date
    [5, r'^(([0-1]?[0-9])|([2][0-3])):([0-5]?[0-9]):([0-5]?[0-9])?$', 'time'],  # time 12:35:29
    [6, r'[\u4e00-\u9fa5]{1,}$', 'Chinese'],  # Chinese
    [11, r'^w+([-+.]w+)*@w+([-.]w+)*.w+([-.]w+)*$', 'email'],  # email
    # [6,r'/^([a-zA-Z0-9]+[_|\_|\.]?)*[a-zA-Z0-9]+@([a-zA-Z0-9]+[_|\_|\.]?)*[a-zA-Z0-9]+\.[a-zA-Z]{2,3}$/'], # Email
    [12, r'^[a-zA-z]+://[^s]*$', 'url'],  # url

    [13, r'((25[0-5]|2[0-4]\d|((1\d{2})|([1-9]?\d)))\.){3}(25[0-5]|2[0-4]\d|((1\d{2})|([1-9]?\d)))', 'IP address'],
    # ip address
    [14,
     r'^[1-9]\d{7}((0\d)|(1[0-2]))(([0|1|2]\d)|3[0-1])\d{3}$|^[1-9]\d{5}[1-9]\d{3}((0\d)|(1[0-2]))(([0|1|2]\d)|3[0-1])\d{3}([0-9]|X)$',
     'ID Card Number'],  # id card
    [15, r'^[0-9a-fA-F]{32}$', 'MD5'],  # md5
    [16, r'^[1][3,4,5,8,7][0-9]{9}$', 'Mobile Phone Number'],  # mobile phone number
    [17,
     r'((\d{11})|^((\d{7,8})|(\d{4}|\d{3})-(\d{7,8})|(\d{4}|\d{3})-(\d{7,8})-(\d{4}|\d{3}|\d{2}|\d{1})|(\d{7,8})-(\d{4}|\d{3}|\d{2}|\d{1}))$)',
     'Telephone Number'],  # telephone number
    [18, r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'url'],
    # [18, r'^(ht|f)tp(s?)\:\/\/[0-9a-zA-Z]([-.\w]*[0-9a-zA-Z])*(:(0-9)*)*(\/?)([a-zA-Z0-9\-\.\?\,\'\/\\\+&amp;%$#_]*)?', 'url'],
    # url
    # [19, r'[A-Z0-9a-z\._\%\+\-]+@[A-Za-z0-9\.\-]+\\.[A-Za-z]{2,4}', 'domain'],
    [19, r'([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,6}', 'domain'],
    # url
    [20, r'(Mozilla)|(AppleWebKit)', 'referer']
]


# if the 3rd column is priority, this function get priority
def getPriorityOfType(typeno):
    priority = -1
    for i in range(0, len(patterns)):
        if patterns[i][0] == typeno:
            priority = patterns[i][2]
            break
    return priority


def getTypeName(typeno):
    """Get the type name"""
    typename = ''
    for i in range(0, len(patterns)):
        if patterns[i][0] == typeno:
            typename = patterns[i][2]
            break
    return typename


# should use json to make muilti levels typs.
# patterns_json = {
#     {0, r'^-?\d+$'},  # num
# }

def getTypes(source):
    # print('source: ', source)
    types = []
    l = len(patterns)
    for i in range(0, l):
        # print("search: ", i, patterns[i][1])
        searchObj = re.search(patterns[i][1], source, re.M | re.I)
        if searchObj:
            # print("found this: ", patterns[i])
            types.append(patterns[i][0])
            # print("searchObj.group() : ", searchObj.group())
            # total = len(searchObj.group())
            # for i1 in range(1, total):
            #     print("searchObj.group() : ", searchObj.group(i1))
    if len(types) == 0:
        if DEBUG:
            print("data type not found!! ", source)
    return types


# 1 retrun: if source be matched more result, to output one of the highest priority.
def getTypeOf(source):
    types = getTypes(source)
    type = 0
    highest = 0
    pos = 0
    if len(types) > 0:
        for i in range(0, len(types)):
            priority = int(types[i] / 10)
            if priority > highest:
                highest = priority
                pos = i
        type = types[pos]
    return type


def isNullRow(row):
    for i in range(0, len(row)):
        if len(row[i].strip()) > 0:
            return False
    return True


@unittest.skip('class skip')
class Test(unittest.TestCase):
    """用于测试字符串解析数据类型"""

    def test_num(self):
        """是否能解析整数"""
        bool1 = getTypes("123")
        self.assertEqual(bool1, [0, 1])

    def test_float(self):
        """是否能解析float"""
        bool1 = getTypes("123.123")
        self.assertEqual(bool1, [1])

    def test_isDatetime(self):
        """是否能解析datetime"""
        bool1 = getTypes("2018-02-14 17:48:30.2008710")
        self.assertEqual(bool1, [3])

    def test_isTime(self):
        """是否能解析time"""
        bool1 = getTypes("17:48:30")
        self.assertEqual(bool1, [4])

    def test_chinease(self):
        """是否能解析Chinese"""
        bool1 = getTypes("中文")
        self.assertEqual(bool1, [6])

    def test_isContainChs(self):
        """是否能解析Chinese and en"""
        bool1 = getTypes("中文en混排")
        self.assertEqual(bool1, [6])

    def test_isIpAddress(self):
        """是否能解析ip"""
        bool1 = getTypes("10.1.205.5")
        self.assertEqual(bool1, [8])

    def test_isIpAddress(self):
        """是否能解析"""
        bool1 = getTypes("10.1.205.5")
        self.assertEqual(bool1, [8])

    def test_is9(self):
        """是否能解析id card"""
        bool1 = getTypes("11010119901010111X")
        self.assertEqual(bool1, [9])

    def test_is10(self):
        """是否能解析mobile phone number"""
        bool1 = getTypes("13900110011")
        self.assertEqual(bool1, [10])

    def test_is12(self):
        """是否能解析telephone number"""
        bool1 = getTypes("010-82325678")
        self.assertEqual(bool1, [12])

    def test_is17(self):
        """是否能解析url"""
        bool1 = getTypes("http://www.asiainfo-sec.com/")
        self.assertEqual(bool1, [17])

    def test_is18(self):
        """是否能解析domain"""
        bool1 = getTypes("www.asiainfo-sec.com/")
        self.assertEqual(bool1, [18])


class TestGetName(unittest.TestCase):
    """用于测试字符串解析数据类型"""

    def test_num(self):
        typeno = 13
        typeName = getTypeName(typeno)
        self.assertEqual(typeName, 'IP address')


class TestType1Return(unittest.TestCase):
    """用于测试字符串解析数据类型"""

    def test_num(self):
        """是否能解析整数"""
        bool1 = getTypeOf("123")
        self.assertEqual(1, bool1)

    def test_float(self):
        """是否能解析float"""
        bool1 = getTypeOf("123.123")
        self.assertEqual(2, bool1)

    def test_isDatetime(self):
        """是否能解析datetime"""
        bool1 = getTypeOf("2018-02-14 17:48:30.2008710")
        self.assertEqual(4, bool1)

    def test_isDate(self):
        """是否能解析date"""
        bool1 = getTypeOf("2018-02-14")
        self.assertEqual(4, bool1)

    def test_isTime(self):
        """是否能解析time"""
        bool1 = getTypeOf("17:48:30")
        self.assertEqual(5, bool1)

    def test_chinease(self):
        """是否能解析Chinese"""
        bool1 = getTypeOf("中文")
        self.assertEqual(6, bool1)

    def test_isContainChs(self):
        """是否能解析Chinese and en"""
        bool1 = getTypeOf("中文en混排")
        self.assertEqual(6, bool1)

    def test_isIpAddress(self):
        """是否能解析ip"""
        bool1 = getTypeOf("10.1.205.5")
        self.assertEqual(13, bool1)

    def test_isIpAddress(self):
        """是否能解析"""
        bool1 = getTypeOf("10.1.205.5")
        self.assertEqual(13, bool1)

    def test_is9(self):
        """是否能解析id card"""
        bool1 = getTypeOf("11010119901010111X")
        self.assertEqual(14, bool1)

    def test_is15(self):
        """是否能解析md5 """
        bool1 = getTypeOf("2EA3E7354323A96CFA7566490714021D")
        self.assertEqual(15, bool1)

    def test_is10(self):
        """是否能解析mobile phone number"""
        bool1 = getTypeOf("13900110011")
        self.assertEqual(16, bool1)

    def test_is12(self):
        """是否能解析telephone number"""
        bool1 = getTypeOf("010-82325678")
        self.assertEqual(17, bool1)

    def test_is18(self):
        """是否能解析url"""
        bool1 = getTypeOf("http://www.asiainfo-sec.com/")
        self.assertEqual(18, bool1)

    def test_is19(self):
        """是否能解析domain"""
        bool1 = getTypeOf("www.asiainfo-sec.com")
        self.assertEqual(19, bool1)

    def test_is20(self):
        """是否能解析domain"""
        bool1 = getTypeOf(
            "Mozilla/5.0 (iPhone; CPU iPhone OS 10_3_3 like Mac OS X) AppleWebKit/603.3.8 (KHTML, like Gecko) Mobile/14G60")
        self.assertEqual(20, bool1)


if __name__ == '__main__':
    unittest.main()

# bool1 = getTypes("2018-02-14 17:48:30.2008710")
# print(bool1)
# bool1 = getTypes("17:48:30.2008710")
# print(bool1)
