from data_rasterize import rasterize
from os.path import join

#方法一：将一个数据的22个序列作为一个list，再进行list的组合，得到的是len==80的list
# def create_dataset(rootpath):
#     data_list = []
#     for i in range(80):
#         data_path = join(rootpath,"{}.xlsx".format(i+1))
#         seq_list = rasterize(data_path)
#         data_list.append(seq_arr)
#     return data_list
#方法二：将一个数据的22个序列作为一个list，再进行list的拼接，得到的是二维list，len==80*22
def create_datalist(rootpath):
    data_list = []
    for i in range(100):
        data_path = join(rootpath, "{}.xlsx".format(i+1))
        seq_list = rasterize(i+1, data_path, False)
        data_list += seq_list
    return data_list

#create train dataset and test dataset
def create_dataset(dataset_num, seq, k):
    train_dat = []
    test_dat = []
    L = len(seq)//dataset_num
    n = 0
    while n < dataset_num*0.8:
        for i in range(L - k):
            indat = seq[(i + n*L):(i + k + n*L)]
            outdat = seq[(i + k + n*L)]
            train_dat.append((indat, [outdat]))
        n += 1
    while dataset_num*0.8 <= n < dataset_num:
        for i in range(L - k):
            indat = seq[(i + n*L):(i + k + n*L)]
            outdat = seq[(i + k + n*L)]
            test_dat.append((indat, [outdat]))
        n += 1
    return train_dat, test_dat