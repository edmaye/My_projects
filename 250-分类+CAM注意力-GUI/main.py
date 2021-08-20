#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-21 14:16:48
# @Author  : Lewis Tian (chtian@hust.edu.cn)
# @Link    : https://lewistian.github.io
# @Version : Python3.7

import time,os,sys,cv2,qdarkstyle,glob
import numpy as np
#import QMutex
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
# model
from process import ProcessThread


imgpath = 'imgs/'

class GUI(QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()
        uic.loadUi("./gui.ui", self)   # 以文件形式加载ui界面

        self.loadImgList()
  
        self.process_thread = ProcessThread()
        self.process_thread.result.connect(self.plot)
        self.process_thread.result_table.connect(self.plot_table)
        

        self.image_btn.clicked.connect(self.detect_img)
        self.allFiles.itemClicked.connect(self.itemClick)  # 列表框关联时间，用信号槽的写法方式不起作用

    
    def loadImgList(self):
        # 加载所有图片名并添加到列表中
        self.allFiles.clear()
        allImgs = glob.glob(imgpath + '*.jpg')
        allImgs += glob.glob(imgpath + '*.png')
        allImgs += glob.glob(imgpath + '*.jpeg')
        allImgs += glob.glob(imgpath + '*.tif')
        for imgTmp in allImgs:
            self.allFiles.addItem(os.path.basename(imgTmp))  # 将此文件添加到列表中

    def plot(self,result):
        """
        将图像显示在QLabel上
        0719: 为了代码复用性，使用字典信息+eval函数，来定位所要绘图的Qlabel
        """
        # 图像内容
        if 'img' in result and 'label' in result:
            img = result['img']
            label = result['label']
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
            eval('self.'+label).setPixmap(QPixmap.fromImage(img))
        # 文字内容
        if 'log' in result:
            txt = result['log']['txt']
            label = result['log']['label']
            eval('self.'+label).setText(txt)
   
    def plot_table(self,predict):
        for i,score in enumerate(predict):
            self.table_predict.setItem(0, i, QTableWidgetItem(str(round(score,4))))#设置j行i列的内容为Value

    # 单击“检测图片按钮”的槽函数
    def detect_img(self):
        if self.process_thread.is_running():
            return
        file = QFileDialog.getOpenFileName(self, '选择图片', '', 'images(*.png; *.jpg; *.jpeg; *.tif);;*')[0]
        if not file:
            return
        self.process_thread.set_imgpath(file)
        self.process_thread.start()



    def itemClick(self):
        if self.process_thread.is_running():
            return
        file = imgpath + self.allFiles.currentItem().text()  #图像的绝对路径
        self.process_thread.set_imgpath(file)
        self.process_thread.start()

        

     


if __name__ == '__main__':
    app = QApplication(sys.argv)
    stylesheet = qdarkstyle.load_stylesheet_pyqt5()
    app.setFont(QFont("微软雅黑", 9))
    app.setWindowIcon(QIcon("icon.ico"))
    app.setStyleSheet(stylesheet)
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())