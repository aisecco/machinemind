import unittest


def IsListSorted_guess(lst):
    listLen = len(lst)
    if listLen <= 1:
        return True
    # 由首个元素和末尾元素猜测可能的排序规则
    if lst[0] == lst[-1]:  # 列表元素相同
        for elem in lst:
            if elem != lst[0]:
                return False
    elif lst[0] < lst[-1]:  # 列表元素升序
        for i, elem in enumerate(lst[1:]):
            if elem < lst[i]:
                return False
    else:  # 列表元素降序
        for i, elem in enumerate(lst[1:]):
            if elem > lst[i]: return False
    return True


def IsListSorted_fastd(lst):
    it = iter(lst)
    try:
        prev = it.__next__()
    except StopIteration:
        return True
    for cur in it:
        if prev > cur:
            return False
        prev = cur
    return True


# 1 - not short; unknown type
# 2 - desc descending order
# 3 - asc ascending order
#  -  other type

def getSortType_fastd(lst):
    ordertype = 0
    it = iter(lst)
    try:
        prev = it.__next__()
    except StopIteration:
        # return True
        ordertype = 1
    for cur in it:
        if prev > cur:
            if ordertype == 0 or ordertype == 2:
                ordertype = 2
            else:
                ordertype = 1
            # return False
        if prev < cur:
            if ordertype == 0 or ordertype == 3:
                ordertype = 3
            else:
                ordertype = 1
        prev = cur
    return ordertype


def getSortType_flow(sorttype, prev, cur):
    if prev > cur:
        if sorttype == 0 or sorttype == 2:
            sorttype = 2
        else:
            sorttype = 1
        # return False
    if prev < cur:
        if sorttype == 0 or sorttype == 3:
            sorttype = 3
        else:
            sorttype = 1
    return sorttype


def IsListSorted_fastk(lst, key=lambda x, y: x <= y):
    it = iter(lst)
    try:
        prev = it.__next__()
    except StopIteration:
        return True
    for cur in it:
        if not key(prev, cur):
            return False
    prev = cur
    return True


def getMinMaxAve_fastd(lst):
    max = 0
    min = 0
    sum = 0
    inx = 0
    ave = 0

    prev = 0
    resultvalue = (min, max, ave)
    it = iter(lst)
    try:
        prev = it.__next__()
        max = prev
        min = prev
        sum = prev
    except StopIteration:
        resultvalue = (min, max, ave)
    for cur in it:
        sum += cur
        inx += 1
        if cur < min:
            min = cur
        if cur > max:
            max = cur

    if (inx > 0):
        ave = sum / inx
    resultvalue = (min, max, ave)
    return resultvalue


class TestGetSortType(unittest.TestCase):
    """用于测试字符串解析数据类型"""

    def test_1(self):
        """type 1"""
        testlist = [10, 2, 3, 5, 9]
        t = getSortType_fastd(testlist)
        self.assertEqual(t, 1)

    def test_2(self):
        """desc"""
        testlist = [10, 9, 8, 5, 1]
        t = getSortType_fastd(testlist)
        self.assertEqual(t, 2)

    def test_3(self):
        """asc"""
        testlist = [1, 2, 3, 5, 9]
        t = getSortType_fastd(testlist)
        self.assertEqual(t, 3)


class TestGetSortTypeFlow(unittest.TestCase):
    """用于测试数据类型"""

    def test_1(self):
        """type 1"""
        testlist = [10, 2, 3, 5, 9]
        sorttype = 0
        it = iter(testlist)
        try:
            prev = it.__next__()
        except StopIteration:
            sorttype = 1

        for cur in it:
            sorttype = getSortType_flow(sorttype, prev, cur)
            prev = cur

        self.assertEqual(1, sorttype)

    def test_2(self):
        """desc"""
        testlist = [10, 9, 8, 5, 1]
        sorttype = 0
        it = iter(testlist)
        try:
            prev = it.__next__()
        except StopIteration:
            sorttype = 1

        for cur in it:
            sorttype = getSortType_flow(sorttype, prev, cur)
            prev = cur
        self.assertEqual(2, sorttype)

    def test_3(self):
        """asc"""
        testlist = [1, 2, 3, 5, 9]
        sorttype = 0
        it = iter(testlist)
        try:
            prev = it.__next__()
        except StopIteration:
            sorttype = 1

        for cur in it:
            sorttype = getSortType_flow(sorttype, prev, cur)
            prev = cur
        self.assertEqual(sorttype, 3)

    def test_getMinMaxAve_fastd(self):
        """asc"""
        testlist = [1, 2, 3, 5, 9, 1000]
        minmaxave = getMinMaxAve_fastd(testlist)

        self.assertEqual(1, minmaxave[0])
        self.assertEqual(1000, minmaxave[1])
        self.assertEqual(204, minmaxave[2])


if __name__ == '__main__':
    unittest.main()
