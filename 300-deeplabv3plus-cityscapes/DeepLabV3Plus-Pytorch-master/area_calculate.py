# -*- coding:utf-8 -*-
import numpy as np
import os,csv
from PIL import Image

# 各类别RGB值
class_colors = [
    #[0,0,0], 
    [128, 64, 128],  
    [244, 35, 232],  
    [70, 70, 70],  
    [102, 102, 156],  
    [190, 153, 153], 
    [153, 153, 153], # 新增的
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0,0,142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]
class_nums = len(class_colors)
class_colors = np.array(class_colors)

def encode_image(img):
    img_encode = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
    # 原本一个像素的值是(r，g，b)，识别其类别并将mask中对应像素的值赋为类别序号（0～10）
    for i in range(class_nums):
        color = class_colors[i,:]
        img_encode[np.where((img==color).all(axis=-1))] = i 
    return img_encode

    

in_rootdir = r"test_results"       # 街景分割结果保存文件夹
out_file = r"pictures-tongji.csv"     # 汇总统计结果保存路径

# 初始值为0，记录当前处理图像数量
count = 0

# 创建csv文件
writer = csv.writer(open(out_file, "w", newline=""), dialect=("excel"))

# 创建表头
writer.writerow(["pid", "road", "sidewalk", "building", "wall", "fence",
                 "pole", "traffic_light", "traffic_sign", "vegetation", "terrain",
                 "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"])

file_set = {}
for file in os.listdir(in_rootdir):
    file_id = file.split('_')[0]
    if file_id in file_set:
        file_set[file_id].append(file)
    else:
        file_set[file_id] = [file]



for file_id in file_set:
    #print('>>>>>>>>>>>><<<<<<<<<<<<<<')
    # 初始化每个灰度值的像素数量为0
    count_dic = {}
    for i in range(class_nums):
        count_dic[i] = 0
    sumcount = 0.0
    # 处理四张图片
    for file in file_set[file_id]: 
        # 获取分割后图像
        img = Image.open(os.path.join(in_rootdir, file))     
        # 记录当前处理的图像数量
        count += 1
        
        ar = encode_image(np.array(img)).flatten().tolist()
        # 计算灰度值在0-18之间的像素数量，并存储到字典中
        for item in count_dic:
            count_dic[item] += ar.count(item)
        # 获得图像中的像素数量
        sumcount += len(ar)
   
    
        # 计算每个灰度的百分比，并写入CSV中
    writer.writerow([file_id, count_dic[0]*1.0/sumcount, count_dic[1]*1.0/sumcount, count_dic[2]*1.0/sumcount,
                    count_dic[3]*1.0/sumcount, count_dic[4] * 1.0 / sumcount, count_dic[5]*1.0/sumcount,
                    count_dic[6]*1.0/sumcount, count_dic[7]*1.0/sumcount, count_dic[8] * 1.0 / sumcount,
                    count_dic[9]*1.0/sumcount, count_dic[10]*1.0/sumcount, count_dic[11]*1.0/sumcount,
                    count_dic[12]*1.0/sumcount, count_dic[13] * 1.0 / sumcount, count_dic[14]*1.0/sumcount,
                    count_dic[15]*1.0/sumcount, count_dic[16]*1.0/sumcount, count_dic[17]*1.0/sumcount,
                    count_dic[18]*1.0/sumcount])
    # 在输出窗口提示统计图像文件名称和当前处理图像数量
    print(file_id, count)


