import pymysql
import pandas as pd
import tools.klinetools as ktool
import tools.datetools as datetool

import unittest

def connect(host='127.0.0.1', port=3306, user='root', password='123456', database='fcdb_pub', charset='utf8'):
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='123456', database='fcdb_pub',
                           charset='utf8')
    return conn

def disconnect( conn ):
    return conn

def getDailyprice(code, startdt, enddt):
    conn = connect()
    cur = conn.cursor(cursor=pymysql.cursors.DictCursor)

    sql = "SELECT * FROM `tq_qt_skdailyprice_m` WHERE `SECODE` = %s AND tradedate >= %s AND tradedate <= %s ORDER BY TRADEDATE "
    # code = '2010000438'

    # values = ['2010000438', '20220501', '20220518']
    values = [code, startdt, enddt]
    res = cur.execute(sql, values)
    rows = cur.fetchall()
    total = len(rows)
    df = None
    if (total > 0):
        df = pd.DataFrame(rows)  #
        # # change the data types
        # # df['TRADEDATE'] = df['TRADEDATE']
        # df['LCLOSE'] = df['LCLOSE'].astype('float')
        # df['TOPEN'] = df['TOPEN'].astype('float')
        # df['TCLOSE'] = df['TCLOSE'].astype('float')
        # df['THIGH'] = df['THIGH'].astype('float')
        # df['TLOW'] = df['TLOW'].astype('float')
        # df['VOL'] = df['VOL'].astype('float')
        # df['AMOUNT'] = df['AMOUNT'].astype('float')
        #
        # df['CHANGE'] = df['CHANGE'].astype('float')
        # df['PCHG'] = df['PCHG'].astype('float')
        # df['AMPLITUDE'] = df['AMPLITUDE'].astype('float')
        # df['NEGOTIABLEMV'] = df['NEGOTIABLEMV'].astype('float')
        # df['TOTMKTCAP'] = df['TOTMKTCAP'].astype('float')
        # df['TURNRATE'] = df['TURNRATE'].astype('float')
        # print(df.dtypes)
    cur.close()
    conn.close()

    return df

def getDataFrom(code, startdt, enddt):
    conn = connect()
    cur = conn.cursor(cursor=pymysql.cursors.DictCursor)

    sql = "SELECT * FROM `tq_qt_skdailyprice_m` WHERE `SECODE` = %s AND tradedate >= %s AND tradedate <= %s ORDER BY TRADEDATE "
    # code = '2010000438'

    # values = ['2010000438', '20220501', '20220518']
    values = [code, startdt, enddt]
    res = cur.execute(sql, values)
    rows = cur.fetchall()
    total = len(rows)
    df = None
    if (total > 0):
        df = pd.DataFrame(rows)  #

        # change the data types
        # df['TRADEDATE'] = df['TRADEDATE']
        df['LCLOSE'] = df['LCLOSE'].astype('float')
        df['TOPEN'] = df['TOPEN'].astype('float')
        df['TCLOSE'] = df['TCLOSE'].astype('float')
        df['THIGH'] = df['THIGH'].astype('float')
        df['TLOW'] = df['TLOW'].astype('float')
        df['VOL'] = df['VOL'].astype('float')
        df['AMOUNT'] = df['AMOUNT'].astype('float')

        df['AVGPRICE'] = df['AVGPRICE'].astype('float')

        df['CHANGE'] = df['CHANGE'].astype('float')
        df['PCHG'] = df['PCHG'].astype('float')
        df['AMPLITUDE'] = df['AMPLITUDE'].astype('float')
        df['NEGOTIABLEMV'] = df['NEGOTIABLEMV'].astype('float')
        df['TOTMKTCAP'] = df['TOTMKTCAP'].astype('float')
        df['TURNRATE'] = df['TURNRATE'].astype('float')

        # print(df.dtypes)
    cur.close()
    conn.close()

    return df

# include startdate and enddate
def getTradedateSet(startdate, endsdate):
    dateset = []
    conn = connect()
    cur = conn.cursor(cursor=pymysql.cursors.DictCursor)
    sql = "SELECT `TRADEDATE` FROM `lt_sk_tradedate` WHERE tradedate >=%s and tradedate <=%s  ORDER BY TRADEDATE"

    values = [startdate, endsdate]
    res = cur.execute(sql, values)
    rows = cur.fetchall()
    total = len(rows)
    df = pd.DataFrame(rows)  #

    for i in range(0, total):
        date = df.loc[i, 'TRADEDATE']
        dateset.append(date)

    cur.close()
    conn.close()

    return dateset

def getData():
    conn = connect()
    cur = conn.cursor(cursor=pymysql.cursors.DictCursor)

    sql = 'SELECT * FROM `fcdb_pub`.`tq_qt_skdailyprice` WHERE `SECODE` = %s order by TRADEDATE'
    code = '2010000438'
    res = cur.execute(sql, code)

    rows = cur.fetchall()
    result_dataframe = pd.DataFrame(rows)  #

    for x in range(result_dataframe.shape[0]):
        ktype = ktool.klinetypeByDataframe(result_dataframe, x)
        print(ktype)
    cur.close()
    conn.close()

def getPrevDate(basedate, days):
    conn = connect()
    cur = conn.cursor(cursor=pymysql.cursors.DictCursor)
    sql = "SELECT `TRADEDATE` FROM `lt_sk_tradedate` WHERE tradedate <%s  ORDER BY TRADEDATE desc limit 0,%s"

    # values = ['2010000438', '20220501', '20220518']
    values = [basedate, days]
    res = cur.execute(sql, values)
    rows = cur.fetchall()
    total = len(rows)

    prevdate = ''
    if total > 0:
        df = pd.DataFrame(rows)  #
        prevdate = df.loc[total-1, 'TRADEDATE']
    else:
        prevdate = datetool.toStr(datetool.getNextWorkday(basedate, days))

    cur.close()
    conn.close()

    return prevdate

def getNextDate(basedate, days=1):
    nextdate = basedate

    conn = connect()
    cur = conn.cursor(cursor=pymysql.cursors.DictCursor)
    sql = "SELECT `TRADEDATE` FROM `lt_sk_tradedate` WHERE tradedate >%s  ORDER BY TRADEDATE limit 0,%s"

    # values = ['2010000438', '20220501', '20220518']
    values = [basedate, days]
    res = cur.execute(sql, values)
    rows = cur.fetchall()
    total = len(rows)

    nextdate = 'NEXT'
    if total > (days -1 ):
        df = pd.DataFrame(rows)  #
        nextdate = df.loc[days-1, 'TRADEDATE']
    else:
        nextdate = datetool.toStr(datetool.getNextWorkday(basedate, days))
    cur.close()
    conn.close()
    return nextdate

class TestDate(unittest.TestCase):
    # """用于测试日期工具"""
    def test_getTradedateSet(self):
        # test NextDate
        ndate1 = getNextDate("20220601")
        print("the next day from database: ", ndate1)
        ndate2 = getNextDate("20220708")
        print("the next day by cal:", ndate2)

        dtset = getTradedateSet('20220601', '20220731')
        print(dtset)

    def test_prevworkday(self):
        print( "previous workday:")
        datestr = '20220719'
        days = 10
        dt1 = datetool.toStr(datetool.getPrevWorkday(datestr,days))

        dataexpect = "20220705"
        print(dataexpect, dt1)
        self.assertEqual(dataexpect, dt1)

    def test_getPrevTradedate(self):
        print( "previous tradedate:")
        datestr = '20220719'
        days = 10
        dt1 = datetool.toStr(getPrevDate(datestr, days))

        dataexpect = "20220705"
        print(dt1, dataexpect)
        self.assertEqual(dt1, dataexpect)

    def test_getNextTradedate(self):
        print( "next tradedate:")
        datestr = '20220701'
        days = 10
        dt1 = datetool.toStr( getNextDate(datestr, days))

        dataexpect = '20220715'
        print(dt1, dataexpect)
        self.assertEqual(dt1, dataexpect)

    def test_getNextTradedate_future(self):
        print("next tradedate:")
        days = 10
        #for future days
        datestr = '20250701'
        dt1 = datetool.toStr(getNextDate(datestr, days))

        dataexpect = '20250715'
        print(dt1, dataexpect)
        self.assertEqual(dt1, dataexpect)

if __name__ == '__main__':
    unittest.main()

