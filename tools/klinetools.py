import pandas as pd
import unittest
import numpy as np


def getSet(result_dataframe, i):
    # x = 0
    # for x in range(result_dataframe.shape[0]):
    #     for y in range(result_dataframe.shape[1]):
    #         print(result_dataframe.iloc[x, y], end=" ")
    #     print(" ")

    LCLOSE = result_dataframe.loc[i, 'LCLOSE']
    TOPEN = result_dataframe.loc[i, 'TOPEN']
    TCLOSE = result_dataframe.loc[i, 'TCLOSE']
    THIGH = result_dataframe.loc[i, 'THIGH']
    TLOW = result_dataframe.loc[i, 'TLOW']
    VOL = result_dataframe.loc[i, 'VOL']
    AMOUNT = result_dataframe.loc[i, 'AMOUNT']

    d = {"LCLOSE": LCLOSE, "TOPEN": TOPEN, "TCLOSE": TCLOSE, "THIGH": THIGH,
         "TLOW": TLOW, "VOL": VOL, "AMOUNT": AMOUNT}
    return d
    # (LCLOSE, TOPEN, TCLOSE, THIGH, TLOW)


def klinetypeByDataframe(result_dataframe, i):
    LCLOSE = result_dataframe.loc[result_dataframe.index[i], 'LCLOSE']
    TOPEN = result_dataframe.loc[result_dataframe.index[i], 'TOPEN']
    TCLOSE = result_dataframe.loc[result_dataframe.index[i], 'TCLOSE']
    THIGH = result_dataframe.loc[result_dataframe.index[i], 'THIGH']
    TLOW = result_dataframe.loc[result_dataframe.index[i], 'TLOW']
    VOL = result_dataframe.loc[result_dataframe.index[i], 'VOL']
    AMOUNT = result_dataframe.loc[result_dataframe.index[i], 'AMOUNT']

    # d = {"LCLOSE": LCLOSE, "TOPEN": TOPEN, "TCLOSE": TCLOSE, "THIGH": THIGH,
    #                                       "TLOW": TLOW, "VOL": VOL, "AMOUNT": AMOUNT}
    return klinetype(LCLOSE, TOPEN, TCLOSE, THIGH, TLOW)


def klinetypeByDataframe9(result_dataframe, i):
    LCLOSE = result_dataframe.loc[result_dataframe.index[i], 'LCLOSE']
    TOPEN = result_dataframe.loc[result_dataframe.index[i], 'TOPEN']
    TCLOSE = result_dataframe.loc[result_dataframe.index[i], 'TCLOSE']
    THIGH = result_dataframe.loc[result_dataframe.index[i], 'THIGH']
    TLOW = result_dataframe.loc[result_dataframe.index[i], 'TLOW']
    VOL = result_dataframe.loc[result_dataframe.index[i], 'VOL']
    AMOUNT = result_dataframe.loc[result_dataframe.index[i], 'AMOUNT']

    # d = {"LCLOSE": LCLOSE, "TOPEN": TOPEN, "TCLOSE": TCLOSE, "THIGH": THIGH,
    #                                       "TLOW": TLOW, "VOL": VOL, "AMOUNT": AMOUNT}
    return klinetype9(LCLOSE, TOPEN, TCLOSE, THIGH, TLOW)


def klinetypeByDataframe21(result_dataframe, base_vol, result_col, i, scale):
    LCLOSE = result_dataframe.loc[result_dataframe.index[i], 'LCLOSE']
    TOPEN = result_dataframe.loc[result_dataframe.index[i], 'TOPEN']
    TCLOSE = result_dataframe.loc[result_dataframe.index[i], 'TCLOSE']
    THIGH = result_dataframe.loc[result_dataframe.index[i], 'THIGH']
    TLOW = result_dataframe.loc[result_dataframe.index[i], 'TLOW']
    VOL = result_dataframe.loc[result_dataframe.index[i], 'VOL']

    AMOUNT = result_dataframe.loc[result_dataframe.index[i], 'AMOUNT']

    base_value = result_dataframe.loc[result_dataframe.index[i], base_vol]
    result_value = result_dataframe.loc[result_dataframe.index[i], result_col]

    # return getTCLOSEScaler(LCLOSE, TOPEN, TCLOSE, scale)
    return getTCLOSEScaler(LCLOSE, base_value, result_value, scale)


def klinetypeByDataframe19(result_dataframe, i):
    LCLOSE = result_dataframe.loc[result_dataframe.index[i], 'LCLOSE']
    TOPEN = result_dataframe.loc[result_dataframe.index[i], 'TOPEN']
    TCLOSE = result_dataframe.loc[result_dataframe.index[i], 'TCLOSE']
    THIGH = result_dataframe.loc[result_dataframe.index[i], 'THIGH']
    TLOW = result_dataframe.loc[result_dataframe.index[i], 'TLOW']
    VOL = result_dataframe.loc[result_dataframe.index[i], 'VOL']
    AMOUNT = result_dataframe.loc[result_dataframe.index[i], 'AMOUNT']

    # d = {"LCLOSE": LCLOSE, "TOPEN": TOPEN, "TCLOSE": TCLOSE, "THIGH": THIGH,
    #                                       "TLOW": TLOW, "VOL": VOL, "AMOUNT": AMOUNT}
    return klinetype19(LCLOSE, TOPEN, TCLOSE, THIGH, TLOW)


def setKType(ktype, newtype):
    # ktype is 2 d, d0-postive d1-ktype1

    # print("################## Found kline type:{0}".format(newtype))
    if ktype[1] > 0:
        print("################## Existed! old:{0}, new:{1}".format(ktype[1], newtype))
        return True

    ktype[1] = newtype
    return False


def foundExitedKType(ktype):
    if ktype > 0:
        return True
    return False


# line type 9
# a simple type method with open close, +/-10% so topen * 10%  is the whole diff
# 0 other
# 9 - 0.98
# 8 - 0.80
# 7 - 0.62
# 6 - 0.38
# 5 - +/- 0.10
# 4   -0.38
# 3   -0.62
# 2   -0.80
# 1   -0.98
def klinetype9(LCLOSE, TOPEN, TCLOSE, THIGH, TLOW):
    ktype = [0, 0]  # 0- positive, 1 - ktype
    positive = -1

    if (TCLOSE - TOPEN >= 0):
        positive = 1
    else:
        positive = 0
    ktype[0] = positive

    # 线实体幅度占最高价最低价幅度的比例
    # 最高和最低差 是 收盘开盘价差 的 倍数
    # diff = top - low
    TRATIO = 0
    # if (THIGH - TLOW) != 0:
    #     TRATIO = abs(TCLOSE - TOPEN) / (THIGH - TLOW)

    # 涨跌幅度
    TSCALE = 0
    if (TOPEN > 0):
        TSCALE = abs(TCLOSE - TOPEN) / TOPEN

    if (positive == 0):
        if (TSCALE >= 0.098):
            setKType(ktype, 1)
        elif (TSCALE > 0.062):
            setKType(ktype, 2)
        elif (TSCALE > 0.038):
            setKType(ktype, 3)
        else:
            setKType(ktype, 4)
    else:
        if (TSCALE >= 0.098):
            setKType(ktype, 9)
        elif (TSCALE > 0.062):
            setKType(ktype, 8)
        elif (TSCALE > 0.038):
            setKType(ktype, 7)
        elif (TSCALE > 0.010):
            setKType(ktype, 6)
        else:  # (TSCALE < 0.010):
            setKType(ktype, 5)

    if (ktype[1] == 0):
        print("Not Found kline type in type 9!")
    return ktype


def klinetype9reverse(klinetypevalue2, TOPEN):
    # request tclose by topen, increase scaler(ktype2)
    TCLOSE = 0
    # 涨跌幅度
    TSCALE = 0
    if (klinetypevalue2 == 0):
        TSCALE = 0
    elif (klinetypevalue2 == 1):
        TSCALE = -0.098
    elif (klinetypevalue2 == 2):
        TSCALE = -0.062
    elif (klinetypevalue2 == 3):
        TSCALE = -0.038
    elif (klinetypevalue2 == 4):
        TSCALE = -0.010
    elif (klinetypevalue2 == 5):
        TSCALE = 0.010
    elif (klinetypevalue2 == 6):
        TSCALE = 0.020
    elif (klinetypevalue2 == 7):
        TSCALE = 0.038
    elif (klinetypevalue2 == 8):
        TSCALE = 0.062
    elif (klinetypevalue2 == 9):
        TSCALE = 0.098
    else:
        TSCALE = 0

    if (TOPEN > 0):
        # TSCALE = abs(TCLOSE - TOPEN) / TOPEN
        TCLOSE = TOPEN + TSCALE * TOPEN

    return TCLOSE


# line type 9
# a simple type method with open close, +/-10% so topen * 10%  is the whole diff

def klinetype19(LCLOSE, TOPEN, TCLOSE, THIGH, TLOW):
    ktype = [0, 0]  # 0- positive, 1 - ktype
    positive = -1

    if (TCLOSE - TOPEN >= 0):
        positive = 1
    else:
        positive = 0
    ktype[0] = positive

    # 线实体幅度占最高价最低价幅度的比例
    # 最高和最低差 是 收盘开盘价差 的 倍数
    # diff = top - low
    TRATIO = 0
    if (THIGH - TLOW) != 0:
        TRATIO = abs(TCLOSE - TOPEN) / (THIGH - TLOW)
    else:
        setKType(ktype, 10)

    # 涨跌幅度
    TSCALE = 0
    if (TOPEN > 0):
        TSCALE = abs(TCLOSE - TOPEN) / TOPEN

    if (positive == 0):
        if (TSCALE >= 0.090):
            setKType(ktype, 1)
        elif (TSCALE > 0.082):
            setKType(ktype, 2)
        elif (TSCALE > 0.070):
            setKType(ktype, 3)
        elif (TSCALE > 0.060):
            setKType(ktype, 4)
        elif (TSCALE > 0.050):
            setKType(ktype, 5)
        elif (TSCALE > 0.040):
            setKType(ktype, 6)
        elif (TSCALE > 0.030):
            setKType(ktype, 7)
        elif (TSCALE > 0.020):
            setKType(ktype, 8)
        elif (TSCALE > 0.010):
            setKType(ktype, 9)
    else:
        if (TSCALE >= 0.090):
            setKType(ktype, 19)
        elif (TSCALE > 0.082):
            setKType(ktype, 18)
        elif (TSCALE > 0.070):
            setKType(ktype, 17)
        elif (TSCALE > 0.060):
            setKType(ktype, 16)
        elif (TSCALE > 0.050):
            setKType(ktype, 15)
        elif (TSCALE > 0.040):
            setKType(ktype, 14)
        elif (TSCALE > 0.030):
            setKType(ktype, 13)
        elif (TSCALE > 0.020):
            setKType(ktype, 12)
        elif (TSCALE > 0.010):
            setKType(ktype, 11)
        else:
            setKType(ktype, 10)

    # setKType(ktype, 1)

    # if (ktype[1] == 0):
    #     print("Not Found kline type!")

    return ktype


def klinetype(LCLOSE, TOPEN, TCLOSE, THIGH, TLOW):
    ktype = [0, 0]  # 0- positive, 1 - ktype
    positive = -1

    if (TCLOSE - TOPEN >= 0):
        positive = 1
    else:
        positive = 0
    ktype[0] = positive

    # 线实体幅度占最高价最低价幅度的比例
    # 最高和最低差 是 收盘开盘价差 的 倍数
    # diff = top - low
    TRATIO = 0
    if (THIGH - TLOW) != 0:
        TRATIO = abs(TCLOSE - TOPEN) / (THIGH - TLOW)

    # 涨跌幅度
    TSCALE = 0
    if (TOPEN > 0):
        TSCALE = abs(TCLOSE - TOPEN) / TOPEN

    if (TRATIO < 0.98 and TSCALE >= 0.0618 and THIGH != TCLOSE and TLOW != TOPEN and (TCLOSE - TOPEN) > 0):
        setKType(ktype, 1)
    if (TRATIO < 0.98 and TSCALE >= 0.0618 and TLOW != TCLOSE and THIGH != TOPEN and (TCLOSE - TOPEN) < 0):
        setKType(ktype, 2)

    # F3 Small Positive:
    if (TRATIO >= 0.382 and TRATIO < 0.98 and TSCALE < 0.0618 and (THIGH != TCLOSE and TLOW != TOPEN) and (
            THIGH - TLOW) > 0 and (TCLOSE - TOPEN) > 0):
        setKType(ktype, 3)
        # ktype = 3
    # F4 Small Nagative:
    if (TRATIO >= 0.382 and TRATIO < 0.98 and TSCALE < 0.0618 and (TLOW != TCLOSE and THIGH != TOPEN) and (
            THIGH - TLOW) > 0 and (TCLOSE - TOPEN) < 0):
        setKType(ktype, 4)

    # F5 Full Positive:
    if (TRATIO == 1 and TSCALE < 0.098 and (THIGH - TLOW) > 0 and (TCLOSE - TOPEN) > 0):
        setKType(ktype, 5)
    # F6 Full Nagative:
    if (TRATIO == 1 and TSCALE < 0.098 and (THIGH - TLOW) > 0 and (TCLOSE - TOPEN) < 0):
        setKType(ktype, 6)

    # F7 TOP CLOSE Positive:
    if (THIGH == TCLOSE and TSCALE >= 0.0382 and TLOW < TOPEN and (THIGH - TLOW) > 0 and (TCLOSE - TOPEN) > 0):
        setKType(ktype, 7)
    # F8 BOOT CLOSE Nagative:
    if (TCLOSE == TLOW and TSCALE >= 0.0382 and THIGH > TOPEN and (THIGH - TLOW) > 0 and (TCLOSE - TOPEN) < 0):
        setKType(ktype, 8)

    # F9 BOOT OPEN Positive:
    if (TLOW == TOPEN and TSCALE >= 0.0382 and THIGH > TCLOSE and (THIGH - TLOW) > 0 and (TCLOSE - TOPEN) > 0):
        setKType(ktype, 9)
    # F10 BOOT OPEN  Nagative:
    if (TOPEN == THIGH and TSCALE >= 0.0382 and TLOW < TCLOSE and (THIGH - TLOW) > 0 and (TCLOSE - TOPEN) < 0):
        setKType(ktype, 10)

    # F11 Small Positive:
    if (TRATIO < 0.25 and (THIGH != TCLOSE and TLOW != TOPEN) and (THIGH - TLOW) > 0 and (TCLOSE - TOPEN) > 0):
        setKType(ktype, 11)
    # F12 Small Nagative:
    if (TRATIO < 0.25 and (TLOW != TCLOSE and THIGH != TOPEN) and (THIGH - TLOW) > 0 and (TCLOSE - TOPEN) < 0):
        setKType(ktype, 12)

    # F13-1 STOP Positive:
    if (THIGH == TCLOSE and TLOW == TOPEN and TSCALE > 0.098):
        setKType(ktype, 13)
    # F13-2 STOP Nagative:
    if (TLOW == TCLOSE and THIGH == TOPEN and (TCLOSE - TOPEN) < 0 and TSCALE > 0.098):
        setKType(ktype, 14)
        # ktype = 13

    # F15  CROSS Positive:
    if (TCLOSE - TLOW != 0 and (THIGH > TCLOSE) and (TCLOSE > TLOW) and (THIGH - TLOW) > 0 and (TCLOSE - TOPEN) == 0):
        setKType(ktype, 15)
        # ktype = 15
    # F16  CROSS Positive:
    if ((THIGH - TCLOSE) > 0 and TLOW == TOPEN and (THIGH - TLOW) > 0 and (TCLOSE - TOPEN) == 0):
        setKType(ktype, 16)
    # F17  CROSS Positive:
    if ((TLOW - TCLOSE) < 0 and THIGH == TOPEN and (THIGH - TLOW) > 0 and (TCLOSE - TOPEN) == 0):
        setKType(ktype, 17)
    # F18  YI LINE Horizon Positive:
    if (TLOW != 0 and TCLOSE != 0 and (THIGH - TLOW) / TLOW < 0.01 and abs(TOPEN - TCLOSE) / TOPEN < 0.01):
        setKType(ktype, 18)

    # F19  umbrella Positive:
    if (THIGH == TCLOSE and TSCALE < 0.0382 and (THIGH - TOPEN) > 0 and (THIGH - TLOW) > 0 and (TCLOSE - TOPEN) > 0):
        setKType(ktype, 19)
        # ktype = 18
    # F20  umbrella Nagative:
    if (TOPEN == THIGH and TSCALE < 0.0382 and (TLOW - TCLOSE) < 0 and (THIGH - TLOW) > 0 and (TCLOSE - TOPEN) < 0):
        setKType(ktype, 20)

    if (ktype[1] == 0):
        print("Not Found kline type 19!")

    return ktype


def setTCLOSE(ktype, index, newclass):
    # ktype is 2 d, d0-postive d1-ktype1 d3 volclass

    # print("################## Found volclass type:{0}".format(volclass))
    if ktype[index] > 0:
        print("################## Existed! old:{0}, new:{1}".format(ktype[1], newclass))
        return True

    ktype[index] = newclass
    return False


TMAX = 0.10
TMIN = -0.10

DO_DEBUG = False

def getTCLOSEScaler(LCLOSE, tbase, tresult, scale):
    TSCALE = -1
    if (tbase > 0):
        TSCALE = (tresult - tbase) / tbase

    # for debug
    ViewDebugInfo = False

    if (TSCALE >= TMAX):
        ViewDebugInfo = True
        if DO_DEBUG:
            print("### THE TSCALE is too big or small: ", TSCALE, tbase, tresult)
        TSCALE = TMAX * 0.98

    if (TSCALE <= TMIN):
        ViewDebugInfo = True
        if DO_DEBUG:
            print("### THE TSCALE is too big or small: ", TSCALE, tbase, tresult)
        TSCALE = TMIN * 1.02

    tdiff = abs(TSCALE - TMIN) / (TMAX - TMIN)

    # tcloseclass = round(tdiff * (scale) )
    tcloseclass = int(tdiff * scale)

    # for debug
    if ViewDebugInfo:
        if DO_DEBUG:
            print("### THE TSCALE has been adjusted to: ", TSCALE, tbase, tresult)
            print("### THEN tcloseclass is: ", tcloseclass)
    #     setVOL(ktype, 2, TVOLSCALE)
    # return ktype
    return tcloseclass


def getTCLOSEOri(tcloseclass, tbase, scale):
    tresult = -1
    tdiff = tcloseclass / scale
    tscale = tdiff * (TMAX - TMIN) + TMIN
    tresult = tbase * (tscale + 1.0)

    return tresult


TOPEN = 6.66
delta = 0.01

scale = 20


class Test_scaler(unittest.TestCase):
    """用于测试"""

    def test_scaler_0(self):
        # 测试极端的情况，如涨幅 >10%
        # TCLOSEExpect = 6.66
        TOPEN = 7.69
        TCLOSEExpect = 8.57
        scale = 20
        scalerExpect = 19
        # TCLOSE = 6.66
        scaler = getTCLOSEScaler(0, TOPEN, TCLOSEExpect, scale)
        print(scaler, scalerExpect)
        self.assertEqual(scaler, scalerExpect)

        TCLOSE = getTCLOSEOri(scaler, TOPEN, scale)

        print(scaler, TOPEN, TCLOSEExpect, TCLOSE, (TCLOSE - TCLOSEExpect), (TCLOSEExpect * delta))

        if (TCLOSEExpect <= TOPEN * (1 + TMAX)):
            self.assertTrue(abs(TCLOSE - TCLOSEExpect) < (TCLOSEExpect * delta))

    def test_scaler_1(self):
        # pre for next case
        for i in range(scale):
            tdiff = TMIN + (TMAX - TMIN) / scale * i
            tcloseclass = TOPEN * (1 + tdiff)
            print(i, tdiff, tcloseclass, np.random.rand())

    def test_scaler(self):
        for i in range(scale + 1):
            scalerExpect = i
            TCLOSEExpect = TOPEN * (1 + TMIN + (TMAX - TMIN) / scale * i)

            TCLOSEExpect = TCLOSEExpect + (np.random.rand() * delta)
            scaler = getTCLOSEScaler(0, TOPEN, TCLOSEExpect, scale)

            print(i, TCLOSEExpect, scaler, scalerExpect, (scaler == scalerExpect))

            self.assertEqual(scaler, scalerExpect)

            TCLOSE = getTCLOSEOri(scaler, TOPEN, scale)

            if (i < scale):
                print("inverse：", TOPEN, TCLOSEExpect, TCLOSE, (TCLOSE - TCLOSEExpect))
                self.assertTrue(abs(TCLOSE - TCLOSEExpect) < (TCLOSEExpect * delta))


class Testklinetype9reverse(unittest.TestCase):

    def test_0(self):
        ktype = 0
        TCLOSEExpect = 6.66

        TCLOSE = klinetype9reverse(ktype, TOPEN)
        print(ktype, TOPEN, TCLOSEExpect, TCLOSE, abs(TCLOSE - TCLOSEExpect), (TCLOSEExpect * delta))
        self.assertTrue(abs(TCLOSE - TCLOSEExpect) < (TCLOSEExpect * delta))

    def test_9(self):
        ktype = 9
        TCLOSEExpect = 7.31

        TCLOSE = klinetype9reverse(ktype, TOPEN)
        print(ktype, TOPEN, TCLOSEExpect, TCLOSE, abs(TCLOSE - TCLOSEExpect), (TCLOSEExpect * delta))
        self.assertTrue(abs(TCLOSE - TCLOSEExpect) < (TCLOSEExpect * delta))

    def test_8(self):
        ktype = 8
        TCLOSEExpect = 7.1928

        TCLOSE = klinetype9reverse(ktype, TOPEN)
        print(ktype, TOPEN, TCLOSEExpect, TCLOSE, abs(TCLOSE - TCLOSEExpect), (TCLOSEExpect * delta))
        self.assertTrue(abs(TCLOSE - TCLOSEExpect) < (TCLOSEExpect * delta))

    def test_7(self):
        ktype = 7
        TCLOSEExpect = 6.993

        TCLOSE = klinetype9reverse(ktype, TOPEN)
        print(ktype, TOPEN, TCLOSEExpect, TCLOSE, abs(TCLOSE - TCLOSEExpect), (TCLOSEExpect * delta))
        self.assertTrue(abs(TCLOSE - TCLOSEExpect) < (TCLOSEExpect * delta))

    def test_6(self):
        ktype = 6
        TCLOSEExpect = 6.7932

        TCLOSE = klinetype9reverse(ktype, TOPEN)
        print(ktype, TOPEN, TCLOSEExpect, TCLOSE, abs(TCLOSE - TCLOSEExpect), (TCLOSEExpect * delta))
        self.assertTrue(abs(TCLOSE - TCLOSEExpect) < (TCLOSEExpect * delta))

    def test_5(self):
        ktype = 5
        TCLOSEExpect = 6.5934

        TCLOSE = klinetype9reverse(ktype, TOPEN)
        print(ktype, TOPEN, TCLOSEExpect, TCLOSE, abs(TCLOSE - TCLOSEExpect), (TCLOSEExpect * delta))
        self.assertTrue(abs(TCLOSE - TCLOSEExpect) < (TCLOSEExpect * delta))

    def test_4(self):
        ktype = 4
        TCLOSEExpect = 6.5268

        TCLOSE = klinetype9reverse(ktype, TOPEN)
        print(ktype, TOPEN, TCLOSEExpect, TCLOSE, abs(TCLOSE - TCLOSEExpect), (TCLOSEExpect * delta))
        self.assertTrue(abs(TCLOSE - TCLOSEExpect) < (TCLOSEExpect * delta))

    def test_3(self):
        ktype = 3
        TCLOSEExpect = 6.327

        TCLOSE = klinetype9reverse(ktype, TOPEN)
        print(ktype, TOPEN, TCLOSEExpect, TCLOSE, abs(TCLOSE - TCLOSEExpect), (TCLOSEExpect * delta))
        self.assertTrue(abs(TCLOSE - TCLOSEExpect) < (TCLOSEExpect * delta))

    def test_2(self):
        ktype = 2
        TCLOSEExpect = 6.1272

        TCLOSE = klinetype9reverse(ktype, TOPEN)
        print(ktype, TOPEN, TCLOSEExpect, TCLOSE, abs(TCLOSE - TCLOSEExpect), (TCLOSEExpect * delta))
        self.assertTrue(abs(TCLOSE - TCLOSEExpect) < (TCLOSEExpect * delta))

    def test_1(self):
        ktype = 1
        TCLOSEExpect = 6.007

        TCLOSE = klinetype9reverse(ktype, TOPEN)
        print(ktype, TOPEN, TCLOSEExpect, TCLOSE, abs(TCLOSE - TCLOSEExpect), (TCLOSEExpect * delta))
        self.assertTrue(abs(TCLOSE - TCLOSEExpect) < (TCLOSEExpect * delta))


if __name__ == '__main__':
    unittest.main()
