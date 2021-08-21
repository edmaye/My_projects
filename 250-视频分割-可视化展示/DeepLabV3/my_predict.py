from network import deeplabv3plus_mobilenet
import os,cv2
import numpy as np
from torch.utils import data
from plot import Cityscapes
from torchvision import transforms as T
import torch
import torch.nn as nn
from PIL import Image



def main():
    decode_fn = Cityscapes.decode_target    # 可视化模块，给预测结果中的不同类别分配颜色，得到一张RGB图
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # 实例化模型
    model = deeplabv3plus_mobilenet(num_classes=19, output_stride=16,pretrained_backbone=False)

    
    # 加载权重
    ckpt_path = 'mobilenet_cityscapes.pth'
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
  
   
 


    # 图像 to Tensor
    transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    # 亮度调整函数
    def imgBrightness(img1, c, b): 
        rows, cols, channels = img1.shape
        blank = np.zeros([rows, cols, channels], img1.dtype)
        rst = cv2.addWeighted(img1, c, blank, 1-c, b)
        return rst
        






    # 主代码
    with torch.no_grad():
        model = model.eval()
        cap = cv2.VideoCapture('video.mp4') 
        
        label = np.zeros((620,1050,3),dtype=np.uint8)   # 用于保存分割图
        output = np.zeros((620,1050*2,3),dtype=np.uint8)    # 输出图：原图像+分割图 拼接到一起，因此宽度加倍


        fps = cap.get(cv2.CAP_PROP_FPS)  # 视频的fps
        width, height = 1050*2, 620     # 保存mp4的图像宽高
        # 保存的mp4文件
        output_video = cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))  # 写入视频

        nums = int(cap.get(7))  # 视频总帧数
        for id in range(nums):
            _, frame = cap.read()
            # 每5帧预测一次分割图（也就是说相邻5帧用同一张分割图，减少运算成本，这里不用5也没事）
            if id % 5==0:
                img = frame[:,110:1160,:]   # 将原视频左右的黑边去掉
                img = imgBrightness(img,0.07,3)     # 调整亮度（调整完后imshow看下，是否和cityscapes数据集亮度差不多）
                img = cv2.resize(img,(2048,1024))   # resize到cityscapes数据集图像尺寸
                
                # 这里不用改
                img = Image.fromarray(img)
                img = transform(img).unsqueeze(0) # To tensor of NCHW
                img = img.to(device)
                # 模型预测，这里不用改

                pred = model(img).max(1)[1].cpu().numpy()[0] # HW
                pred = decode_fn(pred).astype('uint8')
                print(pred.shape)

                # 将预测的分割图，resize回原图大小
                pred = cv2.resize(pred,(1160-110,620))  # 要记得是去掉黑边后的大小
                label = pred
            # 将原图和分割图cat到一起
            output[:,:1050,:] = frame[:,110:1160,:]
            output[:,1050:,:] = label
            # 写入mp4
            output_video.write(output)



            #cv2.imwrite('result/'+str(id)+'.png',output)
            # colorized_preds = Image.fromarray(colorized_preds)
            # if opts.save_val_results_to:
                #colorized_preds.save(os.path.join(opts.save_val_results_to, str(id)+'.png'))

if __name__ == '__main__':
    main()
