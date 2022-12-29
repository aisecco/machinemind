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

source = [1, 2, 3, 4, 5,2, 6, 7, 8, 9]
# source = [9,8,7,6,5,4,3,2,1]
result = IsListSorted_guess(source) # failed
# result = IsListSorted_fastd(source)
print(result)
