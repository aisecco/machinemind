# mode 1, to add more days
# train the many day trade data to 20 types, regression models
# training data is the matrix of frame data before the view point day

# (trade day), this is a super pararmeter, variat, to scale the matrix size
# 100 days
# 200 days
# 300 d
# the matrix size: 32 prars per day

# 1 self : 26 个指标
# 1.0 code
# 1.1 tradedate
# 1.2 lclose, topen, tclose, thigh, tlow, vol, amount


#
# 2 brother：
# 3 huge face():
# 4
# 20220628 预测vol的数据
# 20220630 分类数据采用vol的百分比，（先用100个分类，实际），这样可能会有超出历史vol的预测，可能会有误差。如果可能，使用股票总量？
# 20220702 预测第二天收盘价的输入VOL也采用scaler方法分类，注意：预测的结果还是第二天收盘价

import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pymysql
import tools.klinetools as ktool
import tools.dbtools as db
import tools.datetools as datetool
import tools.classtools as clstool


def show_train_history(train_history, keys):
    plt.title('train history')

    labelstr = ', '.join(keys)
    for key in keys:
        plt.plot(train_history.history[key])  # 训练数据的执行结果

    # plt.plot(train_history.history[loss])  # 验证数据的执行结果
    plt.ylabel(labelstr)
    plt.xlabel('epoch')
    plt.legend(keys, loc='upper right')
    plt.grid(True)  # 显示网格;
    plt.show()


def saveModel(model, evaluate_r):
    model_path = '../../h5bak/{}-{}-{}.h5'.format(SECODE, datetool.getFormated(), str(evaluate_r[1]))
    print('the model has been saved. path: {}'.format(model_path))
    # model = tf.keras.Sequential()
    model.save(model_path)


def loadModel(model_path):
    model = keras.models.load_model(model_path)
    model.summary()

    print(
        'the model has been loaded, now let us test it again! this test will use a new test image set:')
    # testdf2 = getDataFrom(SECODE, TEST_DATE_START2, TEST_DATE_END2)
    # (test_image2, test_label2, test_date2) = buildData(testdf2, SLIDING_SIZE)
    #
    # evaluate_r2 = model.evaluate(test_image2, test_label2)
    # print(evaluate_r2)
    # print(evaluate_r2[1])
    # print(test_date2)

    return model


def getDataFrom(code, startdt, enddt):
    conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='123456', database='fcdb_pub',
                           charset='utf8')
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


# def buildData(df, sliding_size, ispredict=False):
def buildData(df, sliding_size, MAXVOL, MINVOL, VOLSize, ispredict=False):
    _image = []
    _label = []
    _date = []

    if (df.empty):
        return (_image, _label, _date)
    rowstotal = df.shape[0]
    if (rowstotal <= sliding_size):
        print("the data total is too less to create any window!!!!!")
        return (_image, _label, _date)

    _sliding = []
    for i in range(rowstotal):
        if (i >= sliding_size):
            if (len(_sliding) > sliding_size):
                _sliding.pop(0)
            _image.append(_sliding.copy())  # previous data
            # ktype = ktool.klinetypeByDataframe9(df, i)
            # _label.append(ktype[1])
            ktype = ktool.klinetypeByDataframe21(df, "TOPEN", "TCLOSE", i, class_total )
            _label.append(ktype)
            _date.append(df.loc[i, 'TRADEDATE'])  # this date is the predict date!

        # TRADEDATE = dataframe.loc[i, 'TRADEDATE']
        TRADEDATE = df.loc[i, 'TRADEDATE']
        LCLOSE = df.loc[i, 'LCLOSE']
        TOPEN = df.loc[i, 'TOPEN']
        TCLOSE = df.loc[i, 'TCLOSE']
        THIGH = df.loc[i, 'THIGH']
        TLOW = df.loc[i, 'TLOW']
        VOL = df.loc[i, 'VOL']
        AMOUNT = df.loc[i, 'AMOUNT']
        # DEALS = dataframe.loc[i, 'DEALS']

        # try
        LCLOSE = 0.0 #df.loc[i, 'LCLOSE']
        TOPEN = df.loc[i, 'TOPEN']
        TCLOSE = df.loc[i, 'TCLOSE']
        THIGH = 0.0 #df.loc[i, 'THIGH']
        TLOW = 0.0 #df.loc[i, 'TLOW']
        VOL = df.loc[i, 'VOL']
        ktype = clstool.classByDataframe(df, i, MAXVOL, MINVOL, VOLSize)
        VOLScalered = ktype[2]

        AMOUNT = 0 #df.loc[i, 'AMOUNT']

        _image_cell = []
        _image_cell.append(i / 1000)  # 0
        _image_cell.append(LCLOSE)  # 1
        _image_cell.append(TOPEN)  # 2
        _image_cell.append(TCLOSE)  # 3
        _image_cell.append(THIGH)  # 4
        _image_cell.append(TLOW)  # 5
        _image_cell.append(VOLScalered / 1.0)  # 6
        _image_cell.append(AMOUNT / 1.0)  # 7

        # _image_cell.append(df['CHANGE'])  # 8
        # _image_cell.append(df['PCHG'])  # 9
        # _image_cell.append(df['AMPLITUDE'])  # 10
        # _image_cell.append(df['NEGOTIABLEMV'] / 1000.0)  # 11
        # _image_cell.append(df['TOTMKTCAP'] / 10000.0)  # 12
        # _image_cell.append(df['TURNRATE'])  # 13

        _image_cell.append(datetool.getWeekday(TRADEDATE))  # 14
        _image_cell.append(0.001)  # 15

        _sliding.append(_image_cell)

        if (ispredict and i == rowstotal - 1):

            _image=[]
            _label=['']
            _date=[]

            if (len(_sliding) > sliding_size):
                _sliding.pop(0)
            _image.append(_sliding.copy())
            # but has not label
            _date.append('THE NEXT')

    return (_image, _label, _date)


VOLSize = 100
x = 150  # windows size
y = 10  # the atrribute count
SLIDING_SIZE = x  # days, window size, total in a sliding

# class_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# class_labels = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
class_total = len(class_labels)

# LINE_SIZE = 10 # days then, 1 page is : 10 col * 10 rows


EPOCHS_TOTAL = 500
MODELSAVE_ACC_MIN = 0.8

train_image = []
test_image = []
train_label = []
test_label = []

print("start train set...")

TRAIN_DATE_START = '20181201'
TRAIN_DATE_END = '20220610'

TEST_DATE_START = '20211001'
TEST_DATE_END = '20220731'

# TEST_DATE_START = '20211201'
# TEST_DATE_END = '20220530'

# TEST_DATE_START2 = '20211201'
# TEST_DATE_END2 = '20220530'

# SECODE = '2010000438'  # gzmaotai
SECODE = '2010004836'  # fzzq
# SECODE = '2010003376'  # zhaoshang zhengquan

# SECODE = '2010000942'  # shenzhenyeA 深振业Ａ


train_history = None

totaldf = getDataFrom(SECODE, TRAIN_DATE_START, TEST_DATE_END)
# df = pd.concat([traindf,testdf],axis=0)  # 纵向合并
MAXVOL = totaldf.max()['VOL']
MINVOL = totaldf.min()['VOL']
print(TRAIN_DATE_START, TEST_DATE_END, "max and min VOLs:", MAXVOL, MINVOL)


traindf = getDataFrom(SECODE, TRAIN_DATE_START, TRAIN_DATE_END)
if (not traindf.empty):
    (train_image, train_label, train_date) = buildData(traindf, SLIDING_SIZE, MAXVOL, MINVOL, VOLSize)
    if (len(train_image) > 0):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(x, y)))
        model.add(tf.keras.layers.Dense(128, activation='relu'))

        model.add(tf.keras.layers.Dense(class_total, activation='softmax'))
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['acc']
                      )

        cp_callback = keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.00001, patience=50, verbose=0, mode='max',
                                      baseline=None, restore_best_weights=False)

        train_history = model.fit(train_image, train_label, epochs=EPOCHS_TOTAL, callbacks=[cp_callback])

        testTotal = 1
        print('the training ok, now let us test it, total:', testTotal)

        print("start test set...")
        testdf = getDataFrom(SECODE, TEST_DATE_START, TEST_DATE_END)
        if (not testdf.empty):
            (test_image, test_label, test_date) = buildData(testdf, SLIDING_SIZE, MAXVOL, MINVOL, VOLSize)
            if (len(test_image) > 0):
                evaluate_r = model.evaluate(test_image, test_label)
                print(evaluate_r)
                print(evaluate_r[1])

                print(
                    '{} {} {} {} {} {} {} {} {}'.format(SECODE, TRAIN_DATE_START, TRAIN_DATE_END, len(train_label),
                                                        TEST_DATE_START,
                                                        TEST_DATE_END, len(test_label), EPOCHS_TOTAL, evaluate_r[1]))

                if (evaluate_r[1] > MODELSAVE_ACC_MIN):
                    saveModel(model, evaluate_r)

                (test_image, test_label, test_date) = buildData(testdf, SLIDING_SIZE, MAXVOL, MINVOL, VOLSize, True)
                print(SECODE, 'Start predict (test len: {})'.format(len(test_image)))
                y3 = model.predict(test_image, verbose=1)
                # 23/23 [==============================] - 0s 782us/step
                print('predict shape:{}'.format(y3.shape))
                print('date len', len(test_date), 'label len', len(test_label))
                print('test from', test_date[0], 'to', test_date[-1])
                print('all day:', test_date)
                print('predict(only first of {}): {}'.format(len(y3), y3[0]))
                for i3 in range(len(y3) - 1):
                    print(i3, 'date: {}, label :{}, predict: {}, {:2.0f}% :{}'.format(test_date[i3], test_label[i3],
                                                                                      np.argmax(y3[i3]),
                                                                                      100 * np.max(y3[i3]),
                                                                                      test_label[i3] - np.argmax(
                                                                                          y3[i3])))
                i3 = len(y3) -1
                print('The Next of secode:', SECODE)
                print(i3, 'date: {}, label :{}, predict: {}, {:2.0f}%'.format(test_date[i3], 'NONE',
                                                                                  np.argmax(y3[i3]),
                                                                                  100 * np.max(y3[i3])))
                # 100 * np.max(predictions_array),

# show_train_history(train_history, ['acc', 'loss'])
SHOW_VIZ = False
if (SHOW_VIZ):
    print(train_history.history)
    acc = train_history.history['acc']
    loss = train_history.history['loss']

    epochs = range(len(acc))

    fig, ax = plt.subplots()
    ax.plot(epochs, acc, label='Training acc')
    ax.set_xlabel('epochs')
    ax.set_ylabel("Training acc")
    # plt.legend()
    # plt.figure()
    ax2 = ax.twinx()
    ax2.plot(epochs, loss, label='Training loss')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel("Training loss")

    ax2.legend()
    plt.show()
