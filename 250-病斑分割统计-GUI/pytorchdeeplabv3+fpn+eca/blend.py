#coding:utf-8
import os
import cv2
import numpy as np

def add_mask2image_binary(images_path, masks_path, masked_path):
# Add binary masks to images
    for img_item in os.listdir(images_path):
        print(img_item)
        img_path = os.path.join(images_path, img_item)
        img = cv2.imread(img_path)
        mask_path = os.path.join(masks_path, img_item[:-4]+'.png')  # mask是.png格式的，image是.jpg格式的
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 将彩色mask以二值图像形式读取
        masked = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)  #将image的相素值和mask像素值相加得到结果
        cv2.imwrite(os.path.join(masked_path, img_item), masked)
images_path = './img/'
masks_path = './catmask/'
masked_path = './masked/'
add_mask2image_binary(images_path, masks_path, masked_path)

