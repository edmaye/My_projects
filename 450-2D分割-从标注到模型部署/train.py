import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
import os,glob,cv2,time

import albumentations as A
from lib.dataset import MyDataset
from lib.unet import unet_resnet
from lib.utils import diceCoeff
import copy



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 模型存储路径
model_dir = 'result/checkpoint'
def make_dirs():    # 对应文件夹不存在的话就创建文件夹
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

# 损失函数
bce_fn = torch.nn.BCEWithLogitsLoss()
def bce_loss(outputs, targets):
    return bce_fn(outputs, targets)



## 数据增强
def get_aug(p=0.7):
    return A.Compose([
        A.HorizontalFlip(p=0.5),   # flip-x
        A.RandomResizedCrop(512,512,p=0.6),
        A.OneOf([
            A.RandomContrast( limit=0.2),
            A.RandomBrightness( limit=0.2),
            A.RandomGamma(),
        ]),
    ], p=p)    # 0.836


class_num = 3

def train():
    n_save_iter = 200   # 每隔200 iter保存一下模型
    load_iters = 0      # 加载之前训练的模型(指定iter数)
    
    # 实例化模型
    model = unet_resnet('resnext50_32x4d', 3, class_num, False).to(device)
    # 加载训练好的权重
    if(load_iters!=0):
        pre=torch.load(os.path.join('./result/checkpoint',str(load_iters)+'.pth'))
        model.load_state_dict(pre)
    model.train()   # 将模式设置为train（开启dropout、BN层等）
    opt = Adam(model.parameters(), lr=3e-4) # 使用Adam优化器，并设置学习率


    # 加载训练数据
    dataset = MyDataset(tfms=None,n_class=class_num)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    #dataloader_val = DataLoader(MyDataset(tfms=None,n_class=11,train=False), batch_size=1, shuffle=True, num_workers=0)

    # 训练
    iters = load_iters
    while iters < 1000:
        for imgs,masks in dataloader:
            iters += 1
            imgs = imgs.to(device=device, non_blocking=True)
            masks = masks.to(device=device, non_blocking=True)

            # 模型预测
            output = model(imgs)
            # 计算loss
            loss = bce_loss(output, masks)
            # 反向传播
            opt.zero_grad()
            loss.backward()
            opt.step()

            print('iter{%d} ' % (iters) + ' loss= %.5f ' % (loss.item()))

            # # 计算测试集loss和accuracy    
            # if (iters % n_val_iter == 0):
            #     with torch.no_grad():   # 这部分不参与反向传播，因此不用更新梯度
            #         loss_val, accuracy_val = val(copy.deepcopy(model), dataloader_val, bce_loss, diceCoeff)
            #         print('validation-iter{%d} ' % (iters) + ' loss= %.5f ,accuracy= %.5f' % (loss_val, accuracy_val))


            # 保存模型
            if(iters% n_save_iter == 0):
                save_file_name = os.path.join(model_dir, '%d.pth' % iters)
                torch.save(model.state_dict(), save_file_name)



def val(model,dataloader,loss_fn,metrix):
    with torch.no_grad():
        iters = 0
        loss = 0.0
        accuracy = 0.0
        for imgs,masks in dataloader:
            iters += 1
            # cpu转gpu
            imgs = imgs.to(device=device, non_blocking=True)
            masks = masks.to(device=device, non_blocking=True)
            # 模型预测
            output = model(imgs)
            # 计算loss
            loss += loss_fn(output, masks).item()
            accuracy += metrix(output, masks).item()
        loss /= iters
        accuracy /= iters
    return loss,accuracy

if __name__ == "__main__":

    make_dirs()     # 创建需要的文件夹并指定gpu
    train()