import torch,cv2
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np



class MyDataset(Dataset):
    def __init__(self,kind,train=True) -> None:
        super(MyDataset, self).__init__()
        data = []
        # 打开对应txt文件，读取图片路径和标签信息
        with open('data_'+kind+'.txt',mode='r') as f:
            lines = f.readlines()
            for line in lines:
                if line=='\n':
                    continue
                data.append(line.split(','))
        # 训练集/测试集 划分
        if train:   # 训练集    前80%数据
            self.data = data[:int(len(data)*0.8)]
        else:       # 测试集    后20%数据
            self.data = data[int(len(data)*0.8):]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        img_path,label_1,label_2 = self.data[idx]   # 第idx条数据
        img = cv2.imread(img_path)          # 加载图像
        img = cv2.resize(img,(512,512))     # 原图尺寸太大，适当缩小
        img = img/255.0                     # 归一化
        img = np.transpose(img,(2,0,1))     # 交换维度（满足pytorch要求）：h,w,c --> c,h,w
        img = torch.from_numpy(img.astype(np.float32, copy=False))  # 转化Tensor
        label = torch.Tensor([float(label_1),float(label_2)])       # 转化Tensor
        return img,label







if __name__=='__main__':

    dataset = MyDataset()
    

    for i in range(5):
        img,label = dataset[i]
        print(img.shape,label)
