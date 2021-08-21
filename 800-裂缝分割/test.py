import torch
import numpy as np
import os,cv2,time
import glob

from lib.unet import unet_resnet


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_dir = 'result/checkpoint'




def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))





num_class = 3

label_colors = [
    [0,0,0],
    [0,128,0],
    [0,0,128],
]

def test():
    
    # 提取网络结构
    model = unet_resnet('resnext50_32x4d', 3, num_class, False).to(device)

    # 加载训练好的权重
    pth_epoch = 200 # 加载之前训练的模型(指定轮数)
    pre=torch.load(os.path.join('./result/checkpoint',str(pth_epoch)+'.pth'),map_location=torch.device('cpu'))
    model.load_state_dict(pre)
    model.eval()

    # 一张图片一张图片的预测
    fnames = glob.glob('test/*.jpg')
    for fname in fnames:
        print(fname)

        # 加载图片并进行预处理
        src = cv2.imread(fname)    # 加载图片
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)/255.0   # 归一化

        img = img2tensor((src - src.mean()) / src.std())
        img = img.unsqueeze(0)
        img = img.to(device=device, non_blocking=True)
        output = model(img)[0].detach().cpu().numpy()


        # 每个像素点有一个1x3的向量，代表三个类别概率，将此像素点归为概率最大的那个类别
        output_argmax = np.argmax(output,axis=0)

        # 0-2分别代表3个类别，给他们赋予某种颜色，以实现可视化
        img = np.zeros((output.shape[1],output.shape[2],3),np.uint8)
        for i in range(num_class):
            img[np.where(output_argmax == i)] = label_colors[i]

            print(i,'的比例：',np.where(output_argmax == i)[0].shape[0] / (img.shape[0]*img.shape[1]))

        cv2.imwrite('result/test/' + os.path.basename(fname) + '_predict.png', img)


        
           



if __name__ == "__main__":
    test()