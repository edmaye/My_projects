from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import qdarkstyle,time
import numpy as np
import cv2
import lib

class ProcessThread(QThread):
    result = pyqtSignal(np.ndarray)  # 创建一个自定义信号，元组参数
    def __init__(self):
        super(ProcessThread,self).__init__()
        self.img = None
        self.func = None
        self.running = False

    def is_running(self):
        return self.running

    def set_task(self,img,func):
        if self.running:
            return
        self.img = img
        self.func = func

    def run(self):
        self.running = True
        img_res = self.func(self.img)
        self.result.emit(img_res)


class GUI(QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()  # 调用父类的构造函数
        uic.loadUi("./txcl.ui", self)

        self.openimage.clicked.connect(self.open)

        self.process = ProcessThread()
        self.process.result.connect(self.show_result)


    def open(self):
        openfile = QFileDialog.getOpenFileName(self, '选择图片', '.', 'image files(*.jpg , *.png, *.tiff, *.tif)')[0]
        self.img = cv2.imread(openfile)
        print(openfile)
        print(self.img.shape)
        self.show_source(self.img)
        print(openfile)

    def show_source(self,img):
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(img))

        pass
    def show_result(self):
        print('over')
        pass

    def save(self):
        pass


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    stylesheet = qdarkstyle.load_stylesheet_pyqt5()
    app.setFont(QFont("微软雅黑", 9))
    app.setWindowIcon(QIcon("icon.ico"))
    app.setStyleSheet(stylesheet)
    gui = GUI()
    #app.setStyleSheet("#MainWindow{border-image:url(1.jpg);}")
    # app.setStyleSheet("background-image: url(:/jpg/1.jpg);\n"
    #     "font: 12pt \"微软雅黑\";")


    
    gui.show()
    sys.exit(app.exec_())
