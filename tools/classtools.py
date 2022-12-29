import pandas as pd
import unittest


# 建立动态分类


# 通过归一化方法建立分类，找到最低点，最高点，切分100%
def classByDataframe(result_dataframe, i, MAXVOL, MINVOL, scale):
    LCLOSE = result_dataframe.loc[result_dataframe.index[i], 'LCLOSE']
    TOPEN = result_dataframe.loc[result_dataframe.index[i], 'TOPEN']
    TCLOSE = result_dataframe.loc[result_dataframe.index[i], 'TCLOSE']
    THIGH = result_dataframe.loc[result_dataframe.index[i], 'THIGH']
    TLOW = result_dataframe.loc[result_dataframe.index[i], 'TLOW']
    VOL = result_dataframe.loc[result_dataframe.index[i], 'VOL']
    AMOUNT = result_dataframe.loc[result_dataframe.index[i], 'AMOUNT']

    LVOL = result_dataframe.loc[result_dataframe.index[i], 'VOL']
    TVOL = VOL

    return getVolScaler(LCLOSE, TOPEN, TCLOSE, THIGH, TLOW, LVOL, TVOL, MAXVOL, MINVOL, scale)


def setVOL(ktype, volindex, volclass):
    # ktype is 2 d, d0-postive d1-ktype1 d3 volclass

    # print("################## Found volclass type:{0}".format(volclass))
    if ktype[volindex] > 0:
        print("################## Existed! old:{0}, new:{1}".format(ktype[1], volclass))
        return True

    ktype[volindex] = volclass
    return False


# line type 9
# a simple type method with open close, +/-10% so topen * 10%  is the whole diff
# 0 other

def getVolScaler(LCLOSE, TOPEN, TCLOSE, THIGH, TLOW, LVOL, TVOL, MAXVOL, MINVOL, scale):
    ktype = [0, 0, 0]  # 0- positive, 1 - ktype, 2 - vol
    positive = -1

    if (TCLOSE - TOPEN >= 0):
        positive = 1
    else:
        positive = 0
    ktype[0] = positive

    DIFF = MAXVOL - MINVOL
    if (DIFF > 0):
        tdiff = abs(TVOL - MINVOL) / DIFF
        TVOLSCALE = int(tdiff * scale)
        setVOL(ktype, 2, TVOLSCALE)
    return ktype


def getVolOri(TVOLScaled, MAXVOL, MINVOL, scale):
    TVOL = -1
    DIFF = MAXVOL - MINVOL
    if (DIFF > 0):
        tdiff = TVOLScaled / scale
        TVOL = tdiff * DIFF + MINVOL

    return TVOL


LCLOSE = 0
TOPEN = 1
TCLOSE = 2
THIGH = 10
TLOW = 1
LVOL = 1000.0
MINVOL = 100.0
MAXVOL = 10001.0

scale1 = 10

delta = 0.1


class TestGetSortTypeFlow(unittest.TestCase):
    """用于测试"""

    def test_0(self):
        TVOL = 100.0
        class_scalered = getVolScaler(LCLOSE, TOPEN, TCLOSE, THIGH, TLOW, LVOL, TVOL, MAXVOL, MINVOL, scale1)
        self.assertEqual([1, 0, 0], class_scalered)

        # 反算
        TVOL_Ori = getVolOri(class_scalered[2], MAXVOL, MINVOL, scale1)
        print(TVOL, TVOL_Ori, (TVOL * delta))
        self.assertTrue(abs(TVOL - TVOL_Ori) < (TVOL * delta))

    def test_1(self):
        TVOL = 1200

        class_scalered = getVolScaler(LCLOSE, TOPEN, TCLOSE, THIGH, TLOW, LVOL, TVOL, MAXVOL, MINVOL, scale1)
        self.assertEqual([1, 0, 1], class_scalered)

        # 反算
        TVOL_Ori = getVolOri(class_scalered[2], MAXVOL, MINVOL, scale1)
        print(TVOL, TVOL_Ori, (TVOL * delta))
        self.assertTrue(abs(TVOL - TVOL_Ori) < (TVOL * delta))

    def test_2(self):
        TVOL = 2200

        class_scalered = getVolScaler(LCLOSE, TOPEN, TCLOSE, THIGH, TLOW, LVOL, TVOL, MAXVOL, MINVOL, scale1)
        self.assertEqual([1, 0, 2], class_scalered)

        # 反算
        TVOL_Ori = getVolOri(class_scalered[2], MAXVOL, MINVOL, scale1)
        print(TVOL, TVOL_Ori, (TVOL * delta))
        self.assertTrue(abs(TVOL - TVOL_Ori) < (TVOL * delta))

    def test_3(self):
        TVOL = 3200

        class_scalered = getVolScaler(LCLOSE, TOPEN, TCLOSE, THIGH, TLOW, LVOL, TVOL, MAXVOL, MINVOL, scale1)
        self.assertEqual([1, 0, 3], class_scalered)

        # 反算
        TVOL_Ori = getVolOri(class_scalered[2], MAXVOL, MINVOL, scale1)
        print(TVOL, TVOL_Ori, (TVOL * delta))
        self.assertTrue(abs(TVOL - TVOL_Ori) < (TVOL * delta))

    def test_10(self):
        TVOL = MAXVOL

        class_scalered = getVolScaler(LCLOSE, TOPEN, TCLOSE, THIGH, TLOW, LVOL, TVOL, MAXVOL, MINVOL, scale1)
        self.assertEqual([1, 0, 10], class_scalered)

        # 反算
        TVOL_Ori = getVolOri(class_scalered[2], MAXVOL, MINVOL, scale1)
        print(TVOL, TVOL_Ori, (TVOL * delta))
        self.assertTrue(abs(TVOL - TVOL_Ori) < (TVOL * delta))


if __name__ == '__main__':
    unittest.main()
