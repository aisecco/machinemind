import hashlib

print("1", hash(1))
print("1.0", hash(1.0))    # 相同的数值，不同类型，哈希值是一样的
print("abc", hash("abc"))
print("hello world", hash("hello world"))
print("hello world", hash("hello world"))

# 在运行时发现了一个现象：相同字符串在同一次运行时的哈希值是相同的，但是不同次运行的哈希值不同。
# 这是由于Python的字符串hash算法有一个启动时随机生成secret prefix/suffix的机制，存在随机化现象：对同一个字符串输入，不同解释器进程得到的hash结果可能不同。
# 因此当需要做可重现可跨进程保持一致性的hash，需要用到hashlib模块。
# 20220107 为了获得唯一的md5值，每次都重新建立对象md5 = hashlib.md5()


data = "hello world"
data2 = "hello"
data3 = "helloworld"

md5hash = hashlib.md5(data.encode('utf-8'))
print("1.1", data, md5hash.hexdigest())
md5hash = hashlib.md5(data2.encode('utf-8'))
print("2", data2, md5hash.hexdigest())
md5hash = hashlib.md5(data3.encode('utf-8'))
print("3", data3, md5hash.hexdigest())
md5hash = hashlib.md5(data.encode('utf-8'))
print("1.2", data, md5hash.hexdigest())


md5 = hashlib.md5()  # 应用MD5算法

md5.update(data.encode('utf-8'))
print(data, md5.hexdigest())

md5 = hashlib.md5()  # 应用MD5算法
md5.update(data.encode('utf-8'))
print(data, md5.hexdigest())

print("if update md5 object, the result will be different")
md5.update(data.encode('utf-8'))
print(data, md5.hexdigest())

md5.update(data.encode())
print( "default", md5.hexdigest())

md5.update(data.encode())
print( "default", md5.hexdigest())

md5.update(data.encode("gb2312"))
print( "gb2312", md5.hexdigest())

md5.update(data.encode('ascii'))
print( "ascii", md5.hexdigest())

# new object
md5 = hashlib.md5()
md5.update(data.encode('utf-8'))
print("utf-8:", md5.hexdigest())

md5.update(data.encode('utf-8'))
print("utf-8:", md5.hexdigest())

# new object
md5 = hashlib.md5()
md5.update(data.encode('utf-8'))
print("utf-8:", md5.hexdigest())

md5.update(data.encode('utf-8'))
print("utf-8:", md5.hexdigest())