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
# 20220706 预测vol scale: 20的数据
# 20220707 add AVGPRICE

import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pymysql
import tools.klinetools as ktool
import tools.dbtools as db
import tools.datetools as datetool
from tools import klinetools


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


def buildData(df, resultcolname, sliding_size, ispredict=False):
    _image = []
    _label = []
    _date = []
    _topen = []
    _tclose = []

    if (df.empty):
        return (_image, _label, _date, _topen, _tclose)
    rowstotal = df.shape[0]
    if (rowstotal <= sliding_size):
        print("the data total is too less to create any window!!!!!")
        return (_image, _label, _date, _topen, _tclose)

    _sliding = []
    for i in range(rowstotal):
        if (i >= sliding_size):
            if (len(_sliding) > sliding_size):
                _sliding.pop(0)
            _image.append(_sliding.copy())  # previous data
            tclose_scaler = ktool.klinetypeByDataframe21(df, basecolname, resultcolname,  i, class_total)
            if (tclose_scaler > (class_total - 1)):
                print("XXXXX", df.loc[i, 'TRADEDATE'], df.loc[i, basecolname], df.loc[i, resultcolname], tclose_scaler)

            _label.append(tclose_scaler)

            _date.append(df.loc[i, 'TRADEDATE'])  # this date is the predict date!
            _topen.append(df.loc[i, basecolname])  # this base value to calculate the result !
            # _tclose.append(df.loc[i, 'TCLOSE'])  # this date is the predict tclose!
            _tclose.append(df.loc[i, resultcolname])  # try to the highest

        # TRADEDATE = dataframe.loc[i, 'TRADEDATE']
        TRADEDATE = df.loc[i, 'TRADEDATE']
        LCLOSE = df.loc[i, 'LCLOSE']
        TOPEN = df.loc[i, 'TOPEN']
        TCLOSE = df.loc[i, 'TCLOSE']
        THIGH = df.loc[i, 'THIGH']
        TLOW = df.loc[i, 'TLOW']
        VOL = df.loc[i, 'VOL']
        AMOUNT = df.loc[i, 'AMOUNT']
        AVGPRICE = df.loc[i, 'AVGPRICE']
        # DEALS = dataframe.loc[i, 'DEALS']
        CHANGE = df.loc[i, 'CHANGE']
        PCHG = df.loc[i,'PCHG']
        AMPLITUDE = df.loc[i,'AMPLITUDE']
        NEGOTIABLEMV = df.loc[i,'NEGOTIABLEMV']
        TOTMKTCAP = df.loc[i,'TOTMKTCAP']
        TURNRATE = df.loc[i,'TURNRATE']

        # try
        # LCLOSE = df.loc[i, 'LCLOSE'] # basecolname
        # LCLOSE = 0.0
        # TOPEN = 0.0 # df.loc[i, 'TOPEN']
        # TCLOSE = 0.0 # df.loc[i, 'TCLOSE']
        # THIGH = 0.0  # df.loc[i, 'THIGH']
        # TLOW = 0.0  # df.loc[i, 'TLOW']
        # VOL = 0.0 # df.loc[i, 'VOL']
        # AMOUNT = 0  # df.loc[i, 'AMOUNT']

        _image_cell = []
        _image_cell.append(i / 1000)  # 0
        _image_cell.append(LCLOSE/1000)  # 1
        _image_cell.append(TOPEN/1000)  # 2
        _image_cell.append(TCLOSE/1000)  # 3
        _image_cell.append(THIGH/1000)  # 4
        _image_cell.append(TLOW/1000)  # 5
        _image_cell.append(VOL / 1000000.0)  # 6
        _image_cell.append(AMOUNT / 100.0)  # 7
        _image_cell.append(AVGPRICE)  # 8

        _image_cell.append(CHANGE)  # 9
        _image_cell.append(PCHG)  # 10
        _image_cell.append(AMPLITUDE)  # 11
        _image_cell.append(NEGOTIABLEMV / 1000000.0)  # 12
        _image_cell.append(TOTMKTCAP / 10000000.0)  # 13
        _image_cell.append(TURNRATE)  # 14

        _image_cell.append(datetool.getWeekday(TRADEDATE))  # 15
        # _image_cell.append(0.001)  # 15

        _sliding.append(_image_cell)

        if (ispredict and i == rowstotal - 1):
            print ( 'In order to predict next, mock some inputs from last data.', i, rowstotal )
            #clear and only remain 1 frame

            # _image = []
            # _label = []
            # _date = []
            # _topen = []
            # _tclose = []

            if (len(_sliding) > sliding_size):
                _sliding.pop(0)

            _image.append(_sliding.copy())
            # but has not label
            # _date.append('THE NEXT')
            nextDate = db.getNextDate(df.loc[i, 'TRADEDATE'])
            _date.append(nextDate)
            # and not topen, use lclose
            _topen.append(df.loc[i , 'TCLOSE'])  # instead of lclose
            # _tclose.append(0.0)  # instead of lclose
            _tclose.append(df.loc[i , resultcolname])  # instead of lclose

    return (_image, _label, _date, _topen, _tclose)


train_image = []
test_image = []
train_label = []
test_label = []
# LINE_SIZE = 10 # days then, 1 page is : 10 col * 10 rows

EPOCHS_TOTAL = 500
MODELSAVE_ACC_MIN = 0.86


x = 200  # windows size
# y = 10  # the atrribute count
y = 16  # the atrribute count
SLIDING_SIZE = x  # days, window size, total in a sliding

# basecolname = 'LCLOSE'
basecolname = 'TOPEN'

# resultcolname = 'THIGH'
resultcolname = 'TCLOSE'
# resultcolname = 'TOPEN'

# class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# class_labels = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
class_total = len(class_labels)

TRAIN_DATE_START = '20181201'
TRAIN_DATE_END = '20220816'

TEST_DATE_START = '20210601'
TEST_DATE_END = '20220831'

# TEST_DATE_START = '20211201'
# TEST_DATE_END = '20220530'

# TEST_DATE_START2 = '20211201'
# TEST_DATE_END2 = '20220530'

# SECODE = '2010000438'  # gzmaotai
SECODE = '2010004836'  # fzzq
# SECODE = '2010003376'  # zhaoshang zhengquan

# SECODE = '2010000942'  # shenzhenyeA 深振业Ａ
# SECODE = '2010001258' # 000755sxlq 活跃，升幅大
# SECODE = '2010004732' #byd

train_history = None

traindf = getDataFrom(SECODE, TRAIN_DATE_START, TRAIN_DATE_END)
if (not traindf.empty):
    print("Start prepare train set...")
    (train_image, train_label, train_date, train_topen, train_result) = buildData(traindf, resultcolname, SLIDING_SIZE)
    if (len(train_image) > 0):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(x, y)))
        model.add(tf.keras.layers.Dense(256, activation='relu'))

        model.add(tf.keras.layers.Dense(class_total, activation='softmax'))
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['acc']
                      )

        cp_callback = keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.00001, patience=50, verbose=0,
                                                    mode='max',
                                                    baseline=None, restore_best_weights=False)

        print("Start train..., basecolname:" , basecolname, " resultcolname: ", resultcolname)
        train_history = model.fit(train_image, train_label, epochs=EPOCHS_TOTAL, callbacks=[cp_callback])
        # train_history = model.fit(train_image, train_label, epochs=EPOCHS_TOTAL)
        testTotal = 1
        print('Let us test it, total:', testTotal)

        print("start test set...")
        testdf = getDataFrom(SECODE, TEST_DATE_START, TEST_DATE_END)
        if (not testdf.empty):
            (test_image, test_label, test_date, test_topen, test_result) = buildData(testdf, resultcolname, SLIDING_SIZE)
            if (len(test_image) > 0):
                evaluate_r = model.evaluate(test_image, test_label)
                print(evaluate_r)
                print(evaluate_r[1])

                print(
                    '{} train date: {} - {} label: {};\n test date: {} - {} label {} EPOCHS: {} evalute: {}'.format(SECODE, TRAIN_DATE_START, TRAIN_DATE_END, len(train_label),
                                                        TEST_DATE_START,
                                                        TEST_DATE_END, len(test_label), EPOCHS_TOTAL, evaluate_r[1]))

                if (evaluate_r[1] > MODELSAVE_ACC_MIN):
                    print("########## ****** It's perfect! ******* ##############")
                    saveModel(model, evaluate_r)

                (test_image, test_label, test_date, test_topen, test_result) = buildData(testdf, resultcolname, SLIDING_SIZE, True)
                print('\n############## Start predict (secode: {}, COL: {}, size: {})'.format(SECODE,resultcolname, len(test_image)))
                print('test data first:', test_image[0][-1], test_date[0], test_topen[0], test_result[0])

                y3 = model.predict(test_image, verbose=1)
                # 23/23 [==============================] - 0s 782us/step
                print('predict shape:{}'.format(y3.shape))
                print('date len', len(test_date), '; label len', len(test_label))
                print('test from', test_date[0], 'to', test_date[-1])

                print('all day:', test_date)
                print('predict(only first of {}): {}'.format(len(y3), y3[0]))
                for i3 in range(len(y3) - 1):
                    predictktype = np.argmax(y3[i3])
                    topen = test_topen[i3]
                    tresult = test_result[i3]
                    predict_tresult = klinetools.getTCLOSEOri(predictktype, topen, class_total)
                    print(
                        '{:3} date:{}, label:{:2}, predict:{:2}, conf: {:2.1f}, diff:{:3}; base:{:5.2f}, result:{:5.2f}, predict result:{:5.2f}, diff:{:5.2f}'.format(
                            i3,
                            test_date[i3], test_label[i3],
                            predictktype,
                            np.max(y3[i3]),
                            test_label[i3] - predictktype,
                            topen, tresult, predict_tresult, (tresult - predict_tresult)
                        ))
                i3 = len(y3) - 1
                predictktype = np.argmax(y3[i3])
                # topen = test_topen[i3]
                tresult = test_result[i3]
                predict_tresult = klinetools.getTCLOSEOri(predictktype, tresult, class_total)

                print('#SECODE: {}, the predict value of {} will be:( base {})'.format(SECODE, resultcolname, basecolname))
                print('{:3} date: {}, label:{:2}, predict:{:2}, conf: {:2.1f}, base: {:2.2f}, predict result:{:5.2f}'.format(
                    i3,
                    test_date[i3], 'NONE',
                    np.argmax(y3[i3]),
                    np.max(y3[i3]), tresult, predict_tresult))
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
