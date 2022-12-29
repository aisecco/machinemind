import re

print(re.match('www', 'www.runoob.com').span())  # 在起始位置匹配
print(re.match('com', 'www.runoob.com'))

line = "Cats are smarter than dogs"
searchObj = re.search(r'(.*) are (.*?) .*', line, re.M | re.I)
if searchObj:
    print("searchObj.group() : ", searchObj.group())
    print("searchObj.group(1) : ", searchObj.group(1))
    print("searchObj.group(2) : ", searchObj.group(2))
else:
    print("Nothing found!!")

# re.match与re.search的区别
# re.match只匹配字符串的开始，如果字符串开始不符合正则表达式，则匹配失败，函数返回None；而re.search匹配整个字符串，直到找到一个匹配。
line = "Cats are smarter than dogs"

matchObj = re.match(r'dogs', line, re.M | re.I)
if matchObj:
    print("match --> matchObj.group() : ", matchObj.group())
else:
    print("No match!!")

matchObj = re.search(r'dogs', line, re.M | re.I)
if matchObj:
    print("search --> searchObj.group() : ", matchObj.group())
else:
    print("No match!!")

# 检索和替换
# 删除字符串中的 Python注释
phone = "2004-959-559 # 这是一个国外电话号码"
num = re.sub(r'#.*$', "", phone)
print("电话号码是: ", num)

# 数字，整数
regex1 = r'[0-9]+'
# 数字，flost
regex2 = r'[0-9\.]+'

source = "567"
r = regex1
searchObj = re.search(r, source, re.M | re.I)
if searchObj:
    print("searchObj.group() : ", searchObj.group())
    total = len(searchObj.group())
    # for i1 in range(1, total):
    #     print("searchObj.group() : ", searchObj.group(i1))
else:
    print("Nothing found!!")

source = "567.99"
r = regex2
searchObj = re.search(r, source, re.M | re.I)
if searchObj:
    print("searchObj.group() : ", searchObj.group())
    total = len(searchObj.group())
    # for i1 in range(1, total):
    #     print("searchObj.group() : ", searchObj.group(i1))
else:
    print("Nothing found!!")

def get_items_idx(match_obj):
    ret = dict()
    ret["idx"] = []
    ret["words"] = []
    with_words = True
    if match_obj:

        for obj in match_obj:
            item_num = len(obj.groups())
            for idx in range(1, item_num + 1):
                ret["idx"].append(obj.span(idx))
                if with_words:
                    ret["words"].append(obj.group(idx))
    else:
        print("failed!")
    return ret
n = 1
def extr(text):
    # pattern = r'\s.*wrk.*\-t(\d*) -c(\d*) -d(\d*).*\sRunning (.*) test @ (.*)'
    pattern = r'.*wrk.*\-t(\d*) -c(\d*) -d(\d*).*'

    # pattern = r'(.*)latency (.*?) '
    seo = re.search(pattern, text, re.M | re.I)
    if seo:
        print("se : ", seo.lastindex)
        n = 1
        while n <= seo.lastindex:
            print(n, seo.group(n))
            n = n + 1


if __name__ == "__main__":
    text = "One plus two equals to three."
    match_obj = re.finditer(r"(.*?) plus (.*?) equals to (.*).", text, re.M | re.I)
    print(get_items_idx(match_obj))

    text = "wrk -t6 -c100 -d30s --latency http://www.bing.com"
    extr(text)
