# ML2017 FALL hw3 Build Convolution Neural Network

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt

#

def read_data(filename):
    fin = open(filename, 'r')
    data = fin.read().strip('\r\n')
    data = data.replace(',', ' ').split()
    label = (data[0] == 'label')
    data = data[2:]
    data = np.array(data)
    #   extract feature and remove label 
    x_data = np.delete(data, range(0, len(data), 48*48+1), axis=0)
    x_data = x_data.reshape((-1, 48, 48,1)).astype('float')
    #   extract labels
    y_data = data[::48*48+1].astype('int')
    x_data /= 255
    if label:
        return x_data, y_data
    else:
        return x_data

#

x_train, y_train = read_data('train.csv')
x_test = read_data('test.csv')
result = open('result.csv', 'w')
print('id,label', file = result)

x_valid = x_train[-3000:]
x_train = x_train[:-3000]
y_valid = y_train[-3000:]
y_train = y_train[:-3000]

y_test = np.zeros(x_test.shape[0])
y_valid = np_utils.to_categorical(y_valid, 7)
y_train = np_utils.to_categorical(y_train, 7)
y_test = np_utils.to_categorical(y_test,7)

model = Sequential()
model.add(Flatten(input_shape = x_train.shape[1:]))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(1024, activation='relu', kernel_initializer='truncated_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(2048, activation='relu', kernel_initializer='truncated_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', kernel_initializer='truncated_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax', kernel_initializer='truncated_normal'))

# 0.3 0.3 0.5 0.5 0.5 0.5 0.62970
# 0.2 0.2 0.3 0.3 0.4 0.4 0.60908
# 0.3 0.3 0.4 0.4 0.5 0.5 0.61827

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size = 128, epochs = 30, validation_data = (x_valid, y_valid))
#score = model.evaluate(x_valid, y_valid)
#print('Score & Accuracy: %s, %s' %(score[0], score[1]))
y_test = model.predict_classes(x_test)


for i in range(len(y_test)):
    print('%i,%s' %(i, y_test[i]), file = result)
result.close()
