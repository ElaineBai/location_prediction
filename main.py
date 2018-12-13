import torch
import torch.nn as nn
from torch.autograd import *
import torch.optim as optim
from model import LSTM
from dataset import create_datalist, create_dataset
from torch.utils.data import DataLoader
import math
import os
import pandas as pd
from math import log10

def ToVariable(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)

use_gpu = torch.cuda.is_available()
# print(use_gpu)
input_size = 900
output_size = 900
hidden_dim = 2000
num_layer = 4
model = LSTM(input_size, hidden_dim, num_layer, output_size)
loss_function = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)

if use_gpu:
    model = model.cuda()

root_path = "dataset"
data_num = 100
time_step = 5
datalist = create_datalist(root_path)
train_data, test_data = create_dataset(data_num, datalist, time_step)
# print(len(train_data)) #17*80
# print(len(test_data))  #17*20

def train(epoch):
    for step, input_data in enumerate(train_data, 1):
        seq = ToVariable(input_data[0])
        outs = ToVariable(input_data[1])
        if use_gpu:
            seq = seq.cuda()
            outs = outs.cuda()
        optimizer.zero_grad()
        model.hidden = model.init_hidden()
        modout = model(seq)
        loss = loss_function(modout, outs)
        loss.backward()
        optimizer.step()
        print("step{}/epoch{}: Loss: {:.4f}".format(step, epoch, loss.data[0]))

        # if step%17 == 0:
        #     # print(modout)
        #     for i in range(len(modout[0])):
        #         if modout[0][i] < 0:
        #             modout[0][i] = 0
        #         if modout[0][i]%1 >0.3:
        #             modout[0][i] = math.ceil(modout[0][i])
        #         else:
        #             modout[0][i] = math.floor(modout[0][i])
        #     # print(modout)
        #     loss_int = loss_function(modout, outs)
        #     print(loss_int.data[0])

def test(epoch):
    avg_psnr = 0
    for batch in test_data:
        input, target = ToVariable(batch[0]), ToVariable(batch[1])
        if use_gpu:
            input = input.cuda()
            target = target.cuda()

        prediction = model(input)
        for i in range(len(prediction[0])):
            if prediction[0][i] < 0:
                prediction[0][i] = 0
            if prediction[0][i]%1 >0.3:
                prediction[0][i] = math.ceil(prediction[0][i])
            else:
                prediction[0][i] = math.floor(prediction[0][i])
                # print(modout)
        mse = loss_function(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_data)))

def checkpoint(epoch):
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    if not os.path.exists(os.path.join("checkpoints", "try")):
        os.mkdir(os.path.join("checkpoints", "try"))
    model_out_path = "checkpoints/try/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format("checkpoints" + "try"))

nEpochs = 1
for epoch in range(1, nEpochs+1):
    train(epoch)
    # if epoch % 5 == 0:
    #     checkpoint(epoch)

predDat = []
model = model.eval()
for step, data in enumerate(test_data, 1):
    seq = ToVariable(data[0])
    trueVal = ToVariable(data[1])
    if use_gpu:
        seq = seq.cuda()
        trueVal = trueVal.cuda()
    predDat = model(seq)
    for i in range(len(predDat[0])):
        if predDat[0][i] < 0:
            predDat[0][i] = 0
        if predDat[0][i] % 1 > 0.3:
            predDat[0][i] = math.ceil(predDat[0][i])
        else:
            predDat[0][i] = math.floor(predDat[0][i])
    loss_int = loss_function(predDat, trueVal)
    predDat = predDat[-1].data.cpu().numpy()
    print(predDat)
    print(loss_int.data[0])
    dataframe = pd.DataFrame(predDat)
    num_excel = math.ceil(step//17)
    time = step % 17 + 5
    if not os.path.exists("result"):
        os.mkdir("result")
    if not os.path.exists(os.path.join("result", "excel{}".format(num_excel))):
        os.mkdir(os.path.join("result", "excel{}".format(num_excel)))
    dataframe.to_csv("result/excel{}/prediction_900_time{}.csv".format(num_excel, time), index=False, sep=',')
