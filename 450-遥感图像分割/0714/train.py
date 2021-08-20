import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
import os,glob,cv2,time
from albumentations import *
from dataset import MyDataset

from unet import unet_resnet


#torch.cuda.set_device(1)
model_dir = 'result/checkpoint'
def make_dirs():    # 对应文件夹不存在的话就创建文件夹
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

bce_fn = torch.nn.BCEWithLogitsLoss()
def bce_loss(outputs, targets):
    return bce_fn(outputs, targets)






def get_aug(p=0.7):
    return Compose([
        HorizontalFlip(p=0.5),   # flip-x
        VerticalFlip(p=0.4),     # flip-y
        RandomRotate90(p=0.5),   # rorate
        RandomResizedCrop(512,512,p=0.6),
        OneOf([
            RandomContrast( limit=0.2),
            RandomBrightness( limit=0.2),
            RandomGamma(),

        ]),
    ], p=p)    # 0.836




n_save_iter = 200


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

load_iters = 1200 # 加载之前训练的模型(指定轮数)

def train():
    
    # 实例化模型
    model = unet_resnet('resnet34', 3, 6, False).to(device)
    # 加载训练好的权重
    if(load_iters!=0):
        pre=torch.load(os.path.join('./result/checkpoint',str(load_iters)+'.pth'))
        model.load_state_dict(pre)

    # 将模式设置为train（开启dropout、BN层等）
    model.train()
    # 使用Adam优化器，并设置学习率
    opt = Adam(model.parameters(), lr=3e-4)

    # 加载训练数据
    dataset = MyDataset(tfms=get_aug(),n_class=6)
    dataloader = DataLoader(dataset, batch_size=48, shuffle=True, num_workers=2)
    # 开始训练
    iters = load_iters
    while iters < 10000:
        for imgs,masks in dataloader:
            iters += 1
            # cpu转gpu
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
            # 打印并记录损失值
            print('iter{%d} ' % (iters) + ' loss= %.5f ' % (loss.item()))
            with open('log.txt','a') as f:
                f.write(str(iters)+','+str(loss.item())+'\n')

            # 保存模型
            if(iters% n_save_iter == 0):
                save_file_name = os.path.join(model_dir, '%d.pth' % iters)
                torch.save(model.state_dict(), save_file_name)



if __name__ == "__main__":

    make_dirs()     # 创建需要的文件夹并指定gpu
    train()