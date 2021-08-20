#coding:utf-8
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import qdarkstyle
import cv2

from test import Detector



class GUI(QMainWindow):

    def __init__(self):
        super(GUI, self).__init__()  # 调用父类的构造函数
        uic.loadUi("./main.ui", self)

        self.detector = Detector()
        self.buttonImg.clicked.connect(self.loadImg)
        self.buttonVideo.clicked.connect(self.loadVideo)
    def loadImg(self):

        # QFileDialog就是系统对话框的那个类第一个参数是上下文，第二个参数是弹框的名字，第三个参数是开始打开的路径，第四个参数是需要的格式
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', '.', 'Image files(*.jpg *.gif *.png)')
        self.label.setText(fname)
        if fname:
            src = cv2.imread(fname)
            img1,img2 = self.detector.detect(src)
            self.label_1.setPixmap(QPixmap(QtGui.QImage(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).data,
                                      img1.shape[1],
                                      img1.shape[0],
                                      QtGui.QImage.Format_RGB888)))
            self.label_2.setPixmap(QPixmap(QtGui.QImage(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB).data,
                                      img2.shape[1],
                                      img2.shape[0],
                                      QtGui.QImage.Format_RGB888)))
    def loadVideo(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择视频', '.', 'Image files(*.mp4 *.MP4 *.avi)')
        if fname:
            cap = cv2.VideoCapture(fname)
            a = 0
            while cap.isOpened():
                ret, frame = cap.read()
                a += 1
                self.label.setText('第'+str(a)+'帧')
                if ret == 0:
                    break
                img1,img2 = self.detector.detect(frame)
                self.label_1.setPixmap(QPixmap(QtGui.QImage(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).data,
                                        img1.shape[1],
                                        img1.shape[0],
                                        QtGui.QImage.Format_RGB888)))
                self.label_2.setPixmap(QPixmap(QtGui.QImage(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB).data,
                                        img2.shape[1],
                                        img2.shape[0],
                                        QtGui.QImage.Format_RGB888)))
                QApplication.processEvents()



if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    stylesheet = qdarkstyle.load_stylesheet_pyqt5()
    app.setFont(QFont("微软雅黑", 9))
    app.setWindowIcon(QIcon("icon.ico"))
    app.setStyleSheet(stylesheet)
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())


