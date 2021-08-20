import torch
import numpy as np
import os,cv2,time
import glob

from unet import unet_resnet


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_dir = 'result/checkpoint'

def infer_img(img,model):
    stride = 256 + 128
    window = 512
    shape = img.shape
    idx = 0
    for i in np.arange(0, shape[0], stride):
        for j in np.arange(0, shape[0], stride):
            if (i + window > shape[0] or j + window > shape[1]):
                continue



def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))



pth_epoch = 2400 # 加载之前训练的模型(指定轮数)
stride = 256
window = 512
def test():
    
    # 提取网络结构
    model = unet_resnet('resnet34', 3, 6, False).to(device)

    # 加载训练好的权重
    pre=torch.load(os.path.join('./result/checkpoint',str(pth_epoch)+'.pth'),map_location=torch.device('cpu'))
    model.load_state_dict(pre)
    torch.save(model.state_dict(),'unet.pth',_use_new_zipfile_serialization=False)
    model.eval()

    # 待预测图片
    img_dir = 'inference'
    fnames = glob.glob(img_dir+'/*.png')

    n_class = 6
    # 一张图片一张图片的预测
    for fname in fnames:
        print(fname)

        # 加载图片并进行预处理
        src_ = cv2.imread(fname)    # 加载图片
        src_ = cv2.resize(src_,(round(src_.shape[1]/512)*512,round(src_.shape[0]/512)*512)) # 模型要求输入图片的尺寸为512x512，因此resize一下
        shape = src_.shape
        src = cv2.cvtColor(src_, cv2.COLOR_BGR2RGB)/255.0   # 归一化

        # 生成三个大矩阵，用来保存模型推理结果
        merge = np.zeros((n_class,shape[0],shape[1]),dtype=float)
        merge_score = np.zeros((n_class,shape[0],shape[1]), dtype=float)
        score = np.ones((n_class,window,window), dtype=float)
        drop = 64
        # 一张图片很大，不能直接输入模型，因此用一个512x512的滑动窗口，以步长stride滑动，将小块输入模型进行预测
        # 得到这个小块的预测图（512x512），保存在之前生成的大矩阵merge对应位置上
        for i in np.arange(0, shape[0], stride):
            for j in np.arange(0, shape[1], stride):
                if (i + window >= shape[0] or j + window >= shape[1]):
                    continue

                # 将图像从numpy转为Tensor（pytorch要求的的数据格式）
                img = src[i:i+window,j:j+window].copy()         # 截取滑窗内的图像小块（512x512）
                img = img2tensor((img - img.mean())/img.std())  # 标准化处理
                img = img.unsqueeze(0)
                img = img.to(device=device, non_blocking=True)  # 从cpu放到gpu上

                # 将数据输入模型，得到预测的概率图
                output = model(img)  
                output = output[0].detach().cpu().numpy()   # 从Tensor转化为numpy矩阵

                # 将小块预测图 放到 merge大矩阵上，供之后融合
                if abs(i-0)>drop and abs(i-shape[0])>drop and abs(j-0)>drop and abs(j-shape[1])>drop:
                    # 每个小块记录到大矩阵上的时候，只取中间一部分，外围一圈直接丢弃（模型对边缘处分割效果差）
                    merge[:, i + drop:i + window - drop, j + drop:j + window - drop] += output[:, drop:-drop,drop:-drop]
                    merge_score[:, i + drop:i + window - drop, j + drop:j + window - drop] += score[:, drop:-drop,drop:-drop]
                else:
                    merge[:,i:i+window,j:j+window] += output
                    merge_score[:,i:i+window,j:j+window] += score

        # 图片最右侧滑窗可能会漏掉，单独预测一下
        for i in np.arange(0, shape[0], stride):
            if (i + window >= shape[0]):
                continue
            img = src[i:i+window,-512:].copy()
            img = img2tensor((img - img.mean())/img.std())
            img = img.unsqueeze(0)
            img = img.to(device=device, non_blocking=True)
            output = model(img)  # heatmap
            output = output[0].detach().cpu().numpy()#.astype(np.uint8)*255

            merge[:,i:i+window,-512:] += output
            merge_score[:,i:i+window,-512:] += score

        # 图片最下面滑窗可能会漏掉，单独预测一下
        for j in np.arange(0, shape[1], stride):
            if (j + window >= shape[1]):
                continue
            img = src[-512:,j:j+window].copy()
            img = img2tensor((img - img.mean())/img.std())
            img = img.unsqueeze(0)
            img = img.to(device=device, non_blocking=True)
            output = model(img)  # heatmap
            output = output[0].detach().cpu().numpy()#.astype(np.uint8)*255
            merge[:,-512:,j:j+window] += output
            merge_score[:,-512:,j:j+window] += score

        # 最右边最下面的一个小块可能会漏掉，单独预测一下
        img = src[-512:, -512:].copy()
        img = img2tensor((img - img.mean()) / img.std())
        img = img.unsqueeze(0)
        img = img.to(device=device, non_blocking=True)
        output = model(img)  # heatmap
        output = output[0].detach().cpu().numpy()  # .astype(np.uint8)*255
        merge[:, -512:,-512:] += output
        merge_score[:, -512:, -512:] += score

        # 每个像素点的值除以被计算过的次数，得到最终概率
        merge /= merge_score

        # 每个像素点有一个1x6的向量，代表六个类别概率，看哪个类别概率最大（即softmax的作用）
        output_argmax = np.zeros((merge.shape[1],merge.shape[2]))
        output_argmax = np.argmax(merge,axis=0)

        # 0-5分别代表6各类别，给他们赋予某种颜色，以实现可视化
        img = np.zeros((merge.shape[1],merge.shape[2],3),np.uint8)
        img[np.where(output_argmax == 0)] = [200,128,30]
        img[np.where(output_argmax == 1)] = [0, 0, 128]
        img[np.where(output_argmax == 2)] = [0, 128, 0]
        img[np.where(output_argmax == 3)] = [0, 128, 128]
        img[np.where(output_argmax == 4)] = [128, 0, 0]
        img[np.where(output_argmax == 5)] = [128, 0, 128]
        cv2.imwrite(img_dir+'/test/' + os.path.basename(fname) + '.png', img)

        # 生成半透明图
        add_src = cv2.addWeighted(img,0.5,src_,0.5,0)
        cv2.imwrite(img_dir+'/result/'+os.path.basename(fname) + '.png', add_src)
        
           



if __name__ == "__main__":
    test()