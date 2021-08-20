#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-21 14:16:48
# @Author  : Lewis Tian (chtian@hust.edu.cn)
# @Link    : https://lewistian.github.io
# @Version : Python3.7

import time,os,sys,cv2,qdarkstyle,glob,collections
import numpy as np
# GUI部分
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
# 算法部分
from process import ProcessThread


imgpath = 'imgs/'


class GUI_explain(QMainWindow):
    def __init__(self):
        super(GUI_explain, self).__init__()
        uic.loadUi("./explain.ui", self)   # 以文件形式加载ui界面


class GUI(QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()
        uic.loadUi("./gui.ui", self)   # 以文件形式加载ui界面

        self.result_save = collections.deque()
  
        # 开一个线程，用于算法处理
        self.process_thread = ProcessThread()
        self.process_thread.result.connect(self.show_result_wait)    # 显示计算结果（result这个信号量emit时，会调用 save_result 这个函数）
        self.process_thread.image_origin.connect(self.plot_now)    # 显示原图（image_origin这个信号量emit时，会调用 plot_now 这个函数）
        
        # 三个按钮连接对应的响应函数
        self.button_open.clicked.connect(self.detect_img)   # "图库"按钮被按下时，调用detest_img这个函数
        self.button_show.clicked.connect(self.show_result)  # "计算"按钮被按下时，调用show_result这个函数
        self.button_save.clicked.connect(self.save_result)  # "保存"按钮被按下时，调用save_result这个函数
        self.button_explain.clicked.connect(self.explain)  # "保存"按钮被按下时，调用save_result这个函数
        
    

    def plot_now(self,result):
        """ 显示原图（立即显示）"""
        # 显示原图
        img = result['img']
        label = result['label']
        img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        eval('self.'+label).setPixmap(QPixmap.fromImage(img))
        # 清空上一张图的计算结果
        self.label_predict.setPixmap(QPixmap(""))   # 图像清空
        self.label_leaf_area.setText("")        # 文字清空
        self.label_disease_area.setText("")     # 文字清空
        self.label_ratio.setText("")             # 文字清空
        self.label_rate.setText("")             # 文字清空


    def show_result_wait(self,result):
        """ 
        显示计算结果（延迟显示，只保存下来，等到调用show_result函数时才显示）
        """
        self.result_save.append(result) # 将计算结果保存下来，等到按下 "计算" 按钮时再一个个显示


    def show_result(self):
        """
        将线程信号量发射的内容显示在QLabel上
        """
        if len(self.result_save)==0 and self.label_leaf_area.text()=="" :
            msg_box = QMessageBox(QMessageBox.Warning, 'Warning', '请等待几秒至计算完成！')
            msg_box.exec_()
        while self.result_save: # 将已保存的所有计算结果依次显示出来
            result = self.result_save.popleft()     # 弹出队列首个元素，对其进行处理    result是一个字典
            # 显示图像内容
            if 'img' in result and 'label' in result:
                src = result['img']
                label = result['label']
                img = QImage(src.data, src.shape[1], src.shape[0], QImage.Format_RGB888)
                eval('self.'+label).setPixmap(QPixmap.fromImage(img))
                if label=='label_predict':
                    self.img_predict = src
            # 显示文字内容
            if 'log' in result:
                txt = result['log']['txt']
                label = result['log']['label']
                eval('self.'+label).setText(txt)

    # 单击“检测图片按钮”的槽函数
    def detect_img(self):
        if self.process_thread.is_running():
            msg_box = QMessageBox(QMessageBox.Warning, 'Warning', '请等待上一张图片的计算完成')
            msg_box.exec_()
            return
        file = QFileDialog.getOpenFileName(self, '选择图片', '', 'images(*.png; *.jpg; *.jpeg; *.tif);;*')[0]   # 弹窗选择文件路径
        self.img_name = os.path.basename(file)  # 保存图像名，供保存图像时用
        if not file:
            return
        # 清空待显示的计算结果
        self.result_save = collections.deque()
        self.img_predict = None
        # 运行新一轮的计算（开启线程）
        self.process_thread.set_imgpath(file)   # 将图像输入线程
        self.process_thread.start()


   
   

    def save_result(self):
        """
        保存结果图
        """
        try:
            if self.img_predict is not None:    # 结果图存在才保存
                cv2.imwrite('result/'+self.img_name+'.png', self.img_predict)
            else:
                msg_box = QMessageBox(QMessageBox.Warning, 'Warning', '请先“计算”得到结果，再保存！')
                msg_box.exec_()
        except:
            print('error')
     
    def explain(self):
        self.ui = GUI_explain()
        self.ui.show()
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    stylesheet = qdarkstyle.load_stylesheet_pyqt5()
    app.setFont(QFont("微软雅黑", 9))
    app.setWindowIcon(QIcon("icon.ico"))
    app.setStyleSheet(stylesheet)
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())