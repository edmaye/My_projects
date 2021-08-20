import numpy as np
import cv2
import os,glob



# 处理后label图像的保存路径
if not os.path.exists('../data/label'):
    os.makedirs('../data/label')

# 原始label图像的路径
labels = glob.glob('../data/originlabel/*.png')


# label中各类别对应的RGB值（通过get_color.py脚本获取）
label_colors = [
    [0,0,0],
    [0,128,0],
    [0,0,128],
    # [0,0,222],
    # [0,0,255],
    
]
label_colors = np.array(label_colors)

for label_path in labels:
    img_id = os.path.basename(label_path)

    img = cv2.imread(label_path)
    mask = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
    # 原本一个像素的值是(r，g，b)，识别其类别并将mask中对应像素的值赋为类别序号（0～10）
    for i in range(label_colors.shape[0]):
        color = label_colors[i,:]
        roi = cv2.inRange(img, color-10, color+10)  # label颜色存在渐变失真，因此需要指定一个范围来提取（区间二值化），在此范围内的像素都变成255）
        mask[np.where((roi==255))] = i
        # mask[np.where((roi==255).all(axis=-1))] = i   # 如果是RGB图像中取值，则需要指定axis=-1

    cv2.imwrite('../data/label/'+img_id,mask)
