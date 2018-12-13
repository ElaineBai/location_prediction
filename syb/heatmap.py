# Date: 2018/12/7
# -*- coding: utf-8 -*-
# File :热力图_整合.py
# Author:Shen


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# time = [0, 5, 10, 15]  # 时间序号
# time = [0]
file_num = 87  # 预测文件序号
data_raw = pd.read_csv('data_50m/%s_900.csv' % file_num)

delta = file_num - 81
delta = delta * 18  # 22个数据通过前4个预测剩18个
data_pre = pd.read_csv('data_50m/prediction_900_v1.csv')
data_pre = data_pre.values.clip(0)  # 0以下归零

# data_pre[(data_pre > 0.4) & (data_pre < 1)] = 1
# 对边缘数据降低阈值
data_pre = data_pre.round()  # 四舍五入

for i in range(18):
    hot_pic_raw = data_raw.iloc[i + 4].values.reshape(30, 30)
    vmax = np.max(hot_pic_raw)
    hot_pic_pre = data_pre[i + delta].reshape(30, 30)
    vmax_ = np.max(hot_pic_pre)
    if vmax < vmax_:
        vmax = vmax_
    # 热力图
    '''
    Colormap是MATLAB里面用来设定和获取当前色图的函数，可以设置如下色图： 
    hot 从黑平滑过度到红、橙色和黄色的背景色，然后到白色。 
    cool 包含青绿色和品红色的阴影色。从青绿色平滑变化到品红色。 
    gray 返回线性灰度色图。 
    bone 具有较高的蓝色成分的灰度色图。该色图用于对灰度图添加电子的视图。 
    white 全白的单色色图。 
    spring 包含品红和黄的阴影颜色。 
    summer 包含绿和黄的阴影颜色。 
    autumn 从红色平滑变化到橙色，然后到黄色。 
    winter 包含蓝和绿的阴影色。

    interpolation=’nearest’是把相邻的相同的颜色连成片
    '''
    # vmax 统一图例
    plt.figure(figsize=(10, 8), dpi=240)
    plt.subplot(121)
    plt.imshow(hot_pic_raw, cmap=plt.cm.hot_r, vmax=vmax)
    plt.colorbar(cax=None, ax=None, shrink=0.5)
    plt.title("raw")
    plt.subplot(122)
    plt.imshow(hot_pic_pre, cmap=plt.cm.hot_r, vmax=vmax)
    plt.colorbar(cax=None, ax=None, shrink=0.5)
    plt.title("prediction")
    # im = ax.imshow(data, cmap=plt.cm.hot_r)

    plt.savefig('data_50m/img/%s/%s_900_time%s.png' % (file_num, file_num, i), bbox_inches='tight')
