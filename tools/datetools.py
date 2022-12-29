import time
import datetime, timedelta
import math
import unittest

from dateutil.parser import parse


def isVaildDate(self, date):
    try:
        if ":" in date:
            time.strptime(date, "%Y-%m-%d %H:%M:%S")
        else:
            time.strptime(date, "%Y-%m-%d")
        return True
    except:
        return False


def parseDate(dtstr):
    dt = parse(dtstr)
    return dt

def toStr(self, fmt = '%Y%m%d'):
    if type(self).__name__ == datetime.__name__:
        dt = self.strftime(fmt)
    else:
        dt = self
    return dt

def getDateNext(dtstr):
    dt = parse(dtstr)
    dt = dt + datetime.timedelta(days=1)
    return dt

# the previous workday main mode is to fetch the tradedate talbe, this is the alternate, only removed weekday
def getNextWorkday(dtstr, delta=1):
    dt = parse(dtstr)

    for d in range(0, delta):
        dt = dt + datetime.timedelta(days=1)
        while dt.weekday() + 1 > 5:
            dt = dt + datetime.timedelta(days=1)

    return dt

# the previous workday main mode is to fetch the tradedate talbe, this is the alternate, only removed weekday
def getPrevWorkday(dtstr, delta=1):
    print('#Using getPrevWorkday function!')
    dt = parse(dtstr)
    for d in range(0, delta):
        dt = dt - datetime.timedelta(days=1)
        while dt.weekday() + 1 > 5:
            dt = dt - datetime.timedelta(days=1)
    return dt


def Now(fmt=None):
    dt = datetime.datetime.now()
    if (fmt == None):
        now = dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        now = dt.strftime(fmt)
    return now

def getWeekday(dtstr):
    dt = parse(dtstr)
    wd = dt.weekday() + 1
    return wd


# def getFormated():
#     dt = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
#     return dt

def getFormated(fmt=None):
    if fmt == None:
        fmt = '%Y%m%d%H%M%S'
    dt = datetime.datetime.now().strftime(fmt)
    return dt


class TestDate(unittest.TestCase):
    # """用于测试日期工具"""

    def test_0(self):
        struct_time = time.strptime("30 Nov 00", "%d %b %y")
        print("", struct_time)

        t = time.time()

        print(t)  # 原始时间数据
        print(int(t))  # 秒级时间戳
        print(int(round(t * 1000)))  # 毫秒级时间戳
        print(int(round(t * 1000000)))  # 微秒级时间戳

        print("t1")
        t = int(t * 10000)
        print(int(math.log10(t)))
        t1 = t % pow(10, int(math.log10(t))) % pow(10, int(math.log10(t)) - 1) % pow(10, int(math.log10(t)) - 2) % pow(
            10, int(
                math.log10(t)) - 3)
        print(t1)
        dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1))
        print(dt)
        print("")

        t = time.time()

        print(t)  # 原始时间数据
        print(int(t))  # 秒级时间戳
        print(int(round(t * 1000)))  # 毫秒级时间戳
        print(int(round(t * 1000000)))  # 微秒级时间戳

        now = datetime.datetime.now()
        print(now)
        print(Now())

        dt_ms = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')  # 含微秒的日期时间，来源 比特量化
        print(dt_ms)

        print("getFormated ： ", getFormated())
        print("getFormated %Y%m%d%H%M%S： ", getFormated('%Y%m%d'))

        datestr = '20220522'

        print("weekday： ", getWeekday(datestr))

        print("datetime： ", parseDate(datestr))
        print("nextday： ", getDateNext(datestr))

        print("getNextWorkday： ", getNextWorkday(datestr))
        datestr = '20220520'  # friday
        print("getNextWorkday： ", getNextWorkday(datestr))

    def test_nextworkday(self):
        print( "nextworkday:")
        fmt = '%Y%m%d'
        datestr = '20220519'  # sunday
        dt1 = getNextWorkday(datestr)

        print("type(dt1)", type(dt1))
        print("str(dt1)", str(dt1))
        print(type(dt1).__name__)
        if type(dt1).__name__ == datetime.__name__:
            dt1 = dt1.strftime(fmt)

        dataexpect = "20220520"
        print(dt1,dataexpect)
        self.assertEqual(dt1,dataexpect)

        datestr = '20220520'  # friday
        dataexpect = "20220523"
        dt2 = toStr(getNextWorkday(datestr))

        print(dataexpect, dt2)
        self.assertEqual(dt2, dataexpect)

    def test_prevworkday(self):
        print( "previous workday:")
        fmt = '%Y%m%d'
        datestr = '20220719'  # sunday
        days = 10
        dt1 = toStr(getPrevWorkday(datestr,days))

        dataexpect = "20220705"
        print(dt1, dataexpect)
        self.assertEqual(dt1.strftime(fmt), dataexpect)



if __name__ == '__main__':
    unittest.main()
