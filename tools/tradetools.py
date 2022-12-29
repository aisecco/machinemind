import pymysql
import pandas as pd
import tools.klinetools as ktool
import tools.datetools as datetool
import tools.dbtools as dbtool

import unittest


# 初始化
# 建立初始账号
# per session a
# # 期初金额
# amountstart = 0
# amountnow = 0
#
# # 期初量（） 0
# volstart = 0
# volnow = 0

def stcode_fetch(secode):
    conn = dbtool.connect()
    cur = conn.cursor(cursor=pymysql.cursors.DictCursor)

    sql = "SELECT SYMBOL, SETPYE, SESPELL, SENAME from `tq_oa_stcode` where SECODE=%s AND ISVALID "
    values = [secode]
    res = cur.execute(sql, values)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return rows


def trade_session_init(operator, secode, symbol, sename, moneystart, volstart, amountstart,feerate):
    # operator = 'admin'
    session = None
    # secode, symbol, sename = '000888', '000888', '000888'

    obj1 = trade_session_find(operator, secode)
    if (not obj1 and len(obj1) > 0):

        session = obj1
        session_1 = obj1[0]['SESSION']
        # obj1 = trade_session_fetch(operator, session_1)
        secode, symbol, sename, moneystart, moneynow, volstart, volnow, amountstart, amountnow, deals, avgprice, feerate = \
        obj1[0]['SECODE'], obj1[0]['SYMBOL'], obj1[0]['SENAME'], obj1[0]['MONEYSTART'], obj1[0]['MONEYNOW'], obj1[0][
            'VOLSTART'], obj1[0]['VOLNOW'], obj1[0]['AMOUNTSTART'], obj1[0]['AMOUNTNOW'], obj1[0]['DEALS'], obj1[0][
            'AVGPRICE'], obj1[0]['FEERATE']
    else:
        moneystart, moneynow, volstart, volnow, amountstart, amountnow, deals, avgprice, feerate = 10000, 10000, 0, 0, 0, 0, 0, 0, 0.001
        createdate = datetool.Now()
        session_1 = '123456'
        rc = trade_session_create (operator, session_1, secode, symbol, sename, moneystart, moneynow, volstart, volnow,
                           amountstart, amountnow, deals, avgprice, feerate,
                           createdate)
        session = trade_session_fetch(operator, session_1)
    return session

def trade_session_create (operator, session, secode, symbol, sename, moneystart, moneynow, volstart, volnow,
                           amountstart,
                           amountnow, deals, avgprice, feerate,
                           createdate):
    conn = dbtool.connect()
    cur = conn.cursor()

    sql = "INSERT INTO `sm_trade_session` ( SESSION, SECODE, SYMBOL, SENAME, MONEYSTART, MONEYNOW, VOLSTART, VOLNOW, AMOUNTSTART, AMOUNTNOW, DEALS, AVGPRICE, FEERATE, CREATOR, CREATEDATE ) " \
          "VALUES ( %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
    values = [session, secode, symbol, sename, moneystart, moneynow, volstart, volnow, amountstart, amountnow, deals,
              avgprice, feerate, operator, createdate]
    res = cur.execute(sql, values)
    conn.close()
    return res

def trade_session_fetch(operator, sessionid):
    conn = dbtool.connect()
    cur = conn.cursor(cursor=pymysql.cursors.DictCursor)

    sql = "SELECT * from `sm_trade_session` where SESSION=%s and CREATOR=%s"
    values = [sessionid, operator]
    res = cur.execute(sql, values)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return rows

def trade_session_find(operator, secode):
    conn = dbtool.connect()
    cur = conn.cursor(cursor=pymysql.cursors.DictCursor)

    sql = "SELECT * from `sm_trade_session` where SECODE=%s and CREATOR=%s"
    values = [secode, operator]
    res = cur.execute(sql, values)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return rows

def trade_session_remove(operator, sessionid):
    conn = dbtool.connect()
    cur = conn.cursor(cursor=pymysql.cursors.DictCursor)

    sql = "DELETE from `sm_trade_session` where SESSION=%s and CREATOR=%s"
    values = [sessionid, operator]
    res = cur.execute(sql, values)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return rows

def trade_session_update_afteract(operator, session, moneynow, volnow, amountnow, deals, modifydate):
    conn = dbtool.connect()
    cur = conn.cursor()

    sql = "UPDATE `sm_trade_session` SET `MONEYNOW`=%s, `VOLNOW`=%s, `AMOUNTNOW`=%s, `DEALS`=%s, `MODIFIER`=%s, `MODIFYDATE`=%s WHERE SESSION=%s"
    values = [moneynow, volnow, amountnow, deals, operator, modifydate, session]
    res = cur.execute(sql, values)

    # cur.commit()
    cur.close()
    conn.close()

    return

def trade_session_update(operator, session, secode, symbol, sename, moneystart, moneynow, volstart, volnow, amountstart,
                         amountnow, deals, avgprice,
                         createdate):
    conn = dbtool.connect()
    cur = conn.cursor()

    sql = "UPDATE `sm_trade_session` set SECODE=%s, SYMBOL=%s, SENAME=%s, MONEYSTART=%s, MONEYNOW=%s, VOLSTART=%s, VOLNOW=%s, AMOUNTSTART=%s, AMOUNTNOW=%s, DEALS=%s, AVGPRICE=%s, MODIFIER=%s, MODIFYDATE=%s WHERE SESSION=%s"

    values = [secode, symbol, sename, moneystart, moneynow, volstart, volnow, amountstart, amountnow, deals, avgprice,
              operator, createdate, session]
    res = cur.execute(sql, values)

    # cur.commit()
    cur.close()
    conn.close()

    return res

def trade_action(operator, session, secode, tradedate, dealtime, action, price, vol, amount, fee, createdate):
    conn = dbtool.connect()
    cur = conn.cursor()

    symbol, sename = '', ''
    sql = "INSERT INTO `sm_trade_action` ( SESSION, SECODE, SYMBOL, SENAME, TRADEDATE, DEALTIME, ACTION, PRICE, VOL, AMOUNT, FEE, OPERATOR, CREATEDATE ) " \
          "VALUES( %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
    values = [session, secode, symbol, sename, tradedate, dealtime, action, price, vol, amount, fee, operator,
              createdate]
    res = cur.execute(sql, values)

    # cur.commit()
    cur.close()
    conn.close()

    return res


class TestTradeSession(unittest.TestCase):
    # """用于测试日期工具"""
    def test_trade_session(self):
        operator = 'admin'

        session_1 = '88888888'
        secode, symbol, sename, moneystart, moneynow, volstart, volnow, amountstart, amountnow, deals, avgprice, feerate = '000666', '000666', '000666', 10000, 10000, 0, 0, 0, 0, 0, 0, 0.001
        createdate = datetool.Now()

        obj1 = trade_session_fetch(operator, session_1)
        if (not obj1):
            trade_session_create(operator, session_1, secode, symbol, sename, moneystart, moneynow, volstart, volnow,
                               amountstart, amountnow, deals, avgprice, feerate,
                               createdate)
        else:
            print(len(obj1))
            session_2 = obj1[0]['SESSION']

            secode, symbol, sename, moneystart, moneynow, volstart, volnow, amountstart, amountnow, deals, avgprice, feerate = obj1[0]['SECODE'], obj1[0]['SYMBOL'], obj1[0]['SENAME'], obj1[0]['MONEYSTART'], obj1[0]['MONEYNOW'], obj1[0]['VOLSTART'], obj1[0]['VOLNOW'],obj1[0]['AMOUNTSTART'], obj1[0]['AMOUNTNOW'], obj1[0]['DEALS'], obj1[0]['AVGPRICE'], obj1[0]['FEERATE']
            trade_session_update(operator, session_1, secode, symbol, sename, moneystart, moneynow, volstart, volnow,
                                 amountstart, amountnow,
                                 deals, avgprice,
                                 createdate)

        obj2 = trade_session_find(operator, secode)
        self.assertIsNotNone(obj2)

        if (obj2 != None or not obj2.empty):
            print(len(obj2))
            session_2 = obj2[0]['SESSION']
            self.assertEqual(session_1, session_2)

        action = 1
        tradedate, dealtime, price, vol, fee = datetool.Now('%Y%m%d'), createdate, 1.88, 1000, 0.001
        amount = price * vol
        fee = amount * float(feerate)
        trade_action(operator, session_2, secode, tradedate, dealtime, action, price, vol, amount, fee, createdate)

        moneynow = float(moneynow) - (amount + fee)
        volnow = volnow + vol
        deals = deals + 1
        amountnow = price * float(volnow)

        trade_session_update_afteract(operator, session_2, moneynow, volnow, amountnow, deals, createdate)

        # SELL
        action = 2
        tradedate, dealtime, price, vol, fee = datetool.Now('%Y%m%d'), createdate, 1.99, 1000, 0.001
        amount = price * vol
        fee = amount * float(feerate)
        trade_action(operator, session_2, secode, tradedate, dealtime, action, price, vol, amount, fee, createdate)

        moneynow = float(moneynow) + amount - fee
        volnow = volnow - vol
        deals = deals + 1
        amountnow = price * float(volnow)

        trade_session_update_afteract(operator, session_2, moneynow, volnow, amountnow, deals, createdate)

    def test_trade_init(self):
        operator = 'admin'

        secode, symbol, sename = '000888', '000888', '000888'

        obj1 = trade_session_find(operator, secode)
        if (not obj1 and  len(obj1) > 0 ):
            session_1 = obj1[0]['SESSION']
            # obj1 = trade_session_fetch(operator, session_1)
            secode, symbol, sename, moneystart, moneynow, volstart, volnow, amountstart, amountnow, deals, avgprice, feerate = obj1[0]['SECODE'], obj1[0]['SYMBOL'], obj1[0]['SENAME'], obj1[0]['MONEYSTART'], obj1[0]['MONEYNOW'], obj1[0]['VOLSTART'], obj1[0]['VOLNOW'],obj1[0]['AMOUNTSTART'], obj1[0]['AMOUNTNOW'], obj1[0]['DEALS'], obj1[0]['AVGPRICE'], obj1[0]['FEERATE']
        else:
            moneystart, moneynow, volstart, volnow, amountstart, amountnow, deals, avgprice, feerate = 10000, 10000, 0, 0, 0, 0, 0, 0, 0.001
            createdate = datetool.Now()
            session_1 = '123456'
            trade_session_create(operator, session_1, secode, symbol, sename, moneystart, moneynow, volstart, volnow,
                               amountstart, amountnow, deals, avgprice, feerate,
                               createdate)

    def test_trade_session_init (self):
        operator = 'admin'
        secode, symbol, sename = '000999', '000999', '000999'

        moneystart, volstart, amountstart, feerate = 10000, 0, 0, 0.001

        session = trade_session_init(operator, secode, symbol, sename, moneystart, volstart, amountstart,feerate)
        print (session)


if __name__ == '__main__':
    unittest.main()
