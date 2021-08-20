#coding:utf-8
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import qdarkstyle


# 累加求和函数
def get_sum(n):
    return (1+n)*n//2
# 斐波那契数列生成器
def fibonacci():
    a,b = 0,1
    while True:
        yield b
        a, b = b, a + b

class GUI(QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()  # 调用父类的构造函数
        uic.loadUi("./main.ui", self)

        self.button_sum.clicked.connect(self.sum_event)
        self.button_fib.clicked.connect(self.fib_event)
        self.button_reset.clicked.connect(self.fib_reset)

        self.num_fibonacci = fibonacci()

    # 累加求和按钮响应，如果输入的是数字的话，就调用求和函数并显示在label上
    def sum_event(self):
        num = self.line_input.text()
        if num.isdigit():
            self.line_sum.setText(str(get_sum(int(num))))
    # next一下斐波那契数列生成器，显示在label上
    def fib_event(self):
        self.line_fib.setText(str(next(self.num_fibonacci)))
    # 重置斐波那契数列生成器
    def fib_reset(self):
        self.num_fibonacci = fibonacci()
        self.fib_event()



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

