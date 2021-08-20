#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-21 14:16:48
# @Author  : Lewis Tian (chtian@hust.edu.cn)
# @Link    : https://lewistian.github.io
# @Version : Python3.7

import time,os,sys,cv2,json
import numpy as np
import matplotlib.pyplot as plt
#import QMutex
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets


from deeplabv3_gui import uNet





class ProcessThread(QThread):
    # 视频处理线程
    result = pyqtSignal(dict)  # 一副图像完成处理后，以信号量的方式传出去
    image_origin = pyqtSignal(dict)  # 一副图像完成处理后，以信号量的方式传出去
    def __init__(self):
        super(ProcessThread,self).__init__()
        self.model_ok = False
        self.running = False
        self.stop_flag =False

        self.model_init()
        self.model_ok = True
    def model_init(self):
        self.model = uNet()

    def is_running(self):
        return self.running

    def set_imgpath(self,img_path):
        self.img_path = img_path

    def run(self):
        self.running = True
        src = Image.open(self.img_path).convert("RGB")
        src_array = np.array(src)
        # >>>>>>>>>>>>>>   GUI显示-原图 <<<<<<<<<<<<<<<<<<<<<
        self.image_origin.emit({'img':src_array,'label':'label_origin'})


        # >>>>>>>>>>>>>>   GUI显示-分割结果图 <<<<<<<<<<<<<<<<<<<<<
        img,mask = self.model.detect_image(src)
        self.result.emit({'img':cv2.resize(np.array(img).astype(np.uint8),(200,200)),'label':'label_predict'})


        # >>>>>>>>>>>>>>   计算面积比  <<<<<<<<<<<<<<<<<<<<<
        # 提取绿色区域
        r = src_array[:,:,0].astype(np.float16)
        g = src_array[:,:,1].astype(np.float16)
        leaf = (r-g*0.95).astype(np.uint8)
        _,leaf = cv2.threshold(leaf,127,1,cv2.THRESH_BINARY)
        # 剔除面积小的连通域
        pass

        # 计算面积
        leaf_area = leaf.sum()
        disease_area = mask.sum()
        rate = disease_area/float(leaf_area+disease_area)
        print(leaf_area,disease_area,rate)
        self.result.emit({'log':{'txt':str(leaf_area)+'像素', 'label':'label_leaf_area'}})
        self.result.emit({'log':{'txt':str(disease_area)+'像素', 'label':'label_disease_area'}})
        self.result.emit({'log':{'txt':str(round(rate*100,4))+'%', 'label':'label_ratio'}})
     
        if rate==0:
            level = '0级'
        elif rate < 0.05:
            level = '1级'
        elif rate < 0.25:
            level = '3级'
        elif rate < 0.50:
            level = '5级'
        elif rate < 0.75:
            level = '7级'
        elif rate < 1.0:
            level = '9级'
        self.result.emit({'log':{'txt':level, 'label':'label_rate'}})
        #self.result.emit({'log':{'txt':class_txt[int(predict_cla)],'label':'label_classtxt'}})

     


        self.running = False

  
if __name__=='__main__':
    src = Image.open('img/1022.jpg').convert("RGB")
    src_array = np.array(src)


    # >>>>>>>>>>>>>>   计算面积比  <<<<<<<<<<<<<<<<<<<<<
    # 提取绿色区域
    r = src_array[:,:,0].astype(np.float16)
    g = src_array[:,:,1].astype(np.float16)
    leaf = (r-g*0.95).astype(np.uint8)
    _,leaf = cv2.threshold(leaf,127,1,cv2.THRESH_BINARY)
    # 剔除面积小的连通域
    pass

    # 计算面积
    leaf_area = leaf.sum()
    print(leaf_area)

    # cv2.imshow('~ye',ye*255)




    # cv2.waitKey(0)






