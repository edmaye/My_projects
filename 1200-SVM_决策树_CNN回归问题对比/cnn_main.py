import torch,time,os,copy
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
from cnn_dataset import MyDataset
from cnn_network import CNN
import matplotlib.pyplot as plt


def val(model,dataloader,loss_fn):
    """
        模型评估，在指定的数据集上计算损失函数
    """
    model.eval()
    loss_trust = 0.0
    loss_read = 0.0
    num = 0
    with torch.no_grad():
        for img,label in dataloader:
            num = num+1
            # 数据存入gpu
            img = img.cuda()
            label = label.cuda()
            # 模型预测
            output = model(img)
            # 计算loss
            loss_trust += loss_fn(output[:,0], label[:,0]).item()
            loss_read += loss_fn(output[:,1], label[:,1]).item()
    loss_trust /= num
    loss_read /= num
    # 返回两种类型数据各自的MSE指标
    return loss_trust,loss_read





def cnn_process(kind,max_epoch=25):
    """
        模型训练+评估
    """
    # 实例化网络、定义优化器、损失函数
    model = CNN().cuda()
    model.train()
    model = model.cuda()
    opt = Adam(model.parameters(), lr=3e-4) # Adam优化器
    loss_fn = nn.MSELoss()  # 均方根损失

    # 数据集迭代器（训练集和测试集分开，各自生成一个迭代器）
    dataloader_train = DataLoader(MyDataset(kind), batch_size=16, shuffle=True, num_workers=0, drop_last=False)
    dataloader_val = DataLoader(MyDataset(kind,train=False), batch_size=16, shuffle=True, num_workers=0, drop_last=False)

    # 训练过程
    t1 = time.time()
    epoch = 0
    while epoch < max_epoch:
        model.train()
        epoch = epoch+1
        for img,label in dataloader_train:  # 读取一个batch的数据
            # 数据存入gpu
            img = img.cuda()
            label = label.cuda()
            # 模型预测
            output = model(img)
            # 计算loss
            loss = loss_fn(output, label)
            # 反向传播
            opt.zero_grad()
            loss.backward()
            opt.step()

    # 模型评估：计算测试集精度
    t2 = time.time()
    mse_trust, mse_read = val(model,dataloader_val,loss_fn)
    t3 = time.time()

    # 计算耗时
    time_train = t2-t1
    time_predict = t3-t2
    return mse_trust, mse_read, time_train, time_predict







if __name__=='__main__':
    for kind in ['PD','PB','PP']:
        mse_trust, mse_read, time_train, time_predict = cnn_process(kind)
        print('CNN-%s  trust：%f  read：%f  训练耗时: %fs  预测耗时：%fs'%(kind,round(mse_trust,5),round(mse_read,5),round(time_train,7),round(time_predict,7)))

