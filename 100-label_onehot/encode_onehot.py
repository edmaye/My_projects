import numpy as np
import cv2
import os,glob





# label中各类别对应的RGB值（通过get_color.py脚本获取）
class_colors = [
    [0,0,0],
    [127,127,127],
    [255,255,255],  
]
class_nums = len(class_colors)
class_colors = np.array(class_colors)

# 处理后label图像的保存路径
saved_path = 'output'
os.makedirs(saved_path, exist_ok=True)

# 原始label图像的路径
labels = glob.glob('labels/*.png')




for label_path in labels:
    img_id = os.path.basename(label_path)

    img = cv2.imread(label_path)
    mask = np.zeros((img.shape[0],img.shape[1],class_nums),dtype=np.uint8)
    mask_id = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
    # 原本一个像素的值是(r，g，b)，识别其类别并将mask中对应像素的值赋为类别序号（0～10）
    for i in range(class_nums):
        color = class_colors[i,:]
        mask_id[np.where((img==color).all(axis=-1))] = i 
    for i in range(class_nums):
        mask[:,:,i] = mask_id==i
    cv2.imwrite(os.path.join(saved_path,img_id),mask)
