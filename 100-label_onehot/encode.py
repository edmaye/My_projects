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

# 原始label图像的路径
labels = glob.glob('labels/*.png')



saved_path = 'output'   # 结果保存路径
os.makedirs(saved_path, exist_ok=True)

mode = 2    # 若mode设置为1，则将原色彩标签图转码； 若mode为其他数，则解码成色彩图

if mode ==1 :
    for label_path in labels:
        img_id = os.path.basename(label_path)

        img = cv2.imread(label_path)
        mask = np.zeros((img.shape[0],img.shape[1],class_nums),dtype=np.uint8)

        # 原本一个像素的值是(r，g，b)，识别其类别并将mask中对应像素的值赋为类别序号（0～10）
        for i in range(class_nums):
            
            color = class_colors[i,:]
            mask[np.where((img==color).all(axis=-1))] = [i,i,i]   # 如果是RGB图像中取值，则需要指定axis=-1

        cv2.imwrite(os.path.join(saved_path,img_id),mask)
else:
    for label_path in labels:
        img_id = os.path.basename(label_path)

        img = cv2.imread(label_path)
        mask = np.zeros((img.shape[0],img.shape[1],class_nums),dtype=np.uint8)

        # 原本一个像素的值是(r，g，b)，识别其类别并将mask中对应像素的值赋为类别序号（0～10）
        for i in range(class_nums):
            
            color = class_colors[i,:]
            mask[np.where((img==[i,i,i]).all(axis=-1))] = color   # 如果是RGB图像中取值，则需要指定axis=-1

        cv2.imwrite(os.path.join(saved_path,img_id),mask)