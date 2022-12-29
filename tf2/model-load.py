import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import tools.extractiontools as dtextract
import tools.fstools as fs

matrixroot = "/Volumes/macbak/pub/matrix/"

train_total = 100
test_total = 100
epochs_total = 1000

class_labels = []
# class_total = 98

_dr_ = ".dr."
_dc_ = ".dc."


def getdatamatrix(dirs, start, end, train_files_total):
    _image = []
    _label = []
    # class_total = len(files)

    print("Start destroyed data, include ori files")
    # noise
    for i0 in range(0, len(dirs)):
        for i1 in range(0, train_files_total):
            classlabel = i0  # int(dirs[i0])
            subfolder = dirs[i0]

            filepath = fs.getpath(matrixroot + subfolder, subfolder + _dr_ + str(i1))
            if i1 >= start and i1 < end:
                print(filepath)
                _matrix = dtextract.csvfile2matrix(filepath, ",")
                _image.append(_matrix)
                _label.append(classlabel)
    # del column
    # filepath = pathpre + subfolder + "/" + files[i0] + ".delc." + str(i1)
    # if i1 >= start and i1 < end:
    #     print(filepath)
    #     _matrix = dtextract.extract(filepath, "", 0)
    #     _image.append(_matrix)
    #     _label.append(classlabel)

    return (_image, _label)


trainfiles = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
              '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35',
              '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52',
              '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69',
              '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86',
              '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97']


def gettestimages():
    # trainfiles = fs.getsubdirs(matrixroot)
    return getdatamatrix(trainfiles, 18, 20, test_total)


train_image = []
test_image = []
train_label = []
test_label = []

print("start train set...")
# (train_image, train_label) = gettrainimages()
# train_images = (np.array(train_image))
# print("train_images:", train_images.shape)
print("start test set...")
(test_image, test_label) = gettestimages()

model_path = '/Users/mac/dev/tensorflow2/data/2022011101.h5'
# model = tf.keras.Sequential()
# model.save(model_path)

model = keras.models.load_model(model_path)
model.summary()

print('the training ok, now let us test it')
model.evaluate(test_image, test_label)

ti = []
ti.append(test_image[3])
result = model.predict(ti)
ans = np.argmax(result[0], axis=-1)
print('ans:', ans)


# y1 = model.predict(test_image)
# print('predict :'.format(y1))
