# Date: 2018/12/6
# -*- coding: utf-8 -*-
# File :用户位置预测v4_900.py
# Author:Shen
# Date: 2018/12/1
# -*- coding: utf-8 -*-
# File :用户位置预测v2_keras.py
# Author:Shen
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, ConvLSTM2D, Flatten
from keras.callbacks import EarlyStopping
import pandas as pd


def get_batches(n_steps, input_row, input_col):
    batch_size = 1  # 卷积通道数
    data = pd.read_csv('data_50m/%s_900.csv' % 1)
    rows = data.shape[0]

    num = rows - n_steps
    x_train = data[0: n_steps]
    x_train = x_train.values.reshape(batch_size, n_steps, input_row, input_col, 1)
    y_train = data.iloc[n_steps]
    y_train = y_train.values.reshape(batch_size, input_row * input_col)
    for i in range(1, num):
        x = data[i:i + n_steps]
        x = x.values.reshape(batch_size, n_steps, input_row, input_col, 1)
        y = data.iloc[n_steps + i]
        y = y.values.reshape(batch_size, input_row * input_col)
        x_train = np.vstack((x_train, x))
        y_train = np.vstack((y_train, y))
    for id in range(2, 101):
        data = pd.read_csv('data_50m/%s_900.csv' % id)
        rows = data.shape[0]

        num = rows - n_steps

        for i in range(num):
            x = data[i:i + n_steps]
            x = x.values.reshape(batch_size, n_steps, input_row, input_col, 1)
            y = data.iloc[n_steps + i]
            y = y.values.reshape(batch_size, input_row * input_col)
            x_train = np.vstack((x_train, x))
            y_train = np.vstack((y_train, y))
    return x_train, y_train


def get_pre(n_steps, input_row, input_col):
    # 得到预测数据
    batch_size = 1
    data = pd.read_csv('data_50m/%s_900.csv' % 81)
    rows = data.shape[0]

    num = rows - n_steps
    x_val = data[0: n_steps]
    x_val = x_val.values.reshape(batch_size, n_steps, input_row, input_col, 1)
    y_val = data.iloc[n_steps]
    y_val = y_val.values.reshape(batch_size, input_row * input_col)
    for i in range(1, num):
        x = data[i:i + n_steps]
        x = x.values.reshape(batch_size, n_steps, input_row, input_col, 1)
        y = data.iloc[n_steps + i]
        y = y.values.reshape(batch_size, input_row * input_col)
        x_val = np.vstack((x_val, x))
        y_val = np.vstack((y_val, y))
    for id in range(82, 101):
        data = pd.read_csv('data_50m/%s_900.csv' % id)
        rows = data.shape[0]

        num = rows - n_steps

        for i in range(num):
            x = data[i:i + n_steps]
            x = x.values.reshape(batch_size, n_steps, input_row, input_col, 1)
            y = data.iloc[n_steps + i]
            y = y.values.reshape(batch_size, input_row * input_col)
            x_val = np.vstack((x_val, x))
            y_val = np.vstack((y_val, y))
    return x_val, y_val


data_row = 30
data_col = 30
time_steps = 4  # 根据之前4个时间序列预测
batch_size = 64
x_train, y_train = get_batches(time_steps, data_row, data_col)
x_val, y_val = get_pre(time_steps, data_row, data_col)
model = Sequential()
model.add(ConvLSTM2D(filters=1, kernel_size=(3, 3),
                     padding='same', return_sequences=True, input_shape=(time_steps, data_row, data_col, 1)))
model.add(ConvLSTM2D(filters=1, kernel_size=(3, 3),
                     padding='same', return_sequences=True))
model.add(Flatten())
model.add(Dense(900))

model.compile(loss='mse', optimizer='rmsprop')
# model.load('./model_900_v1.h5')
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x_train, y_train,
          batch_size=batch_size, epochs=30,
          validation_split=0.2, callbacks=[early_stopping])
# patience：2 连续2轮损失不下降则停止训练 validation_split:0.2 20%做验证集
y_prediction = model.predict(x_val)
# model.save('./model_900_v1.h5')
dataframe = pd.DataFrame(y_prediction)
dataframe.to_csv("data_50m/prediction_900_v1.csv", index=False, sep=',')
