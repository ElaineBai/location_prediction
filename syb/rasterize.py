# Date: 2018/11/22
# -*- coding: utf-8 -*-
# File :栅格化.py
# Author:Shen
import pandas as pd
import numpy as np

radius = 25  # 每个栅格尺寸
slide = int(1500 / radius)  # 栅格数目
for file_index in range(1, 101):
    # 导入数据原始数据文件
    data_xls = pd.read_excel('data/%s.xlsx' % file_index, header=None)
    rows = data_xls.shape[0]
    cols = data_xls.shape[1]
    data = np.zeros((rows, int(cols / 3)))
    for i in range(int(cols / 6)):
        data[:, 2 * i] = data_xls.iloc[:, 6 * i + 3] / radius
        data[:, 2 * i + 1] = data_xls.iloc[:, 6 * i + 4] / radius

    rows = data.shape[0]
    cols = data.shape[1]
    hot_pic = np.zeros((rows, slide, slide))
    for n in range(rows):
        for i in range(int(cols / 2)):
            temp_x = data[n, 2 * i] + slide / 2
            if (temp_x > (slide - 1)):
                temp_x = slide - 1
            if (temp_x < 0):
                temp_x = 0
            x = int(temp_x)
            temp_y = data[n, 2 * i + 1] + slide / 2
            if (temp_y > (slide - 1)):
                temp_y = slide - 1
            if (temp_y < 0):
                temp_y = 0
            y = int(temp_y)
            hot_pic[n, x, y] += 1
    # 排成一列
    hot_pic = hot_pic.reshape(rows, -1)

    dataframe = pd.DataFrame(hot_pic)
    dataframe.to_csv('data_%sm/%s_%s.csv' % (radius, file_index, slide * slide), index=False, sep=',')
    print(file_index, ':done')
