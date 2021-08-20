from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import qdarkstyle,time
import numpy as np
from pyqtgraph import PlotWidget
import pyqtgraph as pg
import cv2,os
from collections import deque
from database import Database
import pandas as pd
from xpinyin import Pinyin
pinyin = Pinyin()

class GUI(QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()  # 调用父类的构造函数
        # 加载ui文件
        uic.loadUi("./main.ui", self)

        # 生成绘图板
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.plot_widget = PlotWidget(self,title="绘图板")  # 这里设置标题，也直接把title删掉，就没有标题了
        self.plot_obj = self.plot_widget.plot(pen=pg.mkPen(color=(0,191,255),width=4))
        self.canvas.addWidget(self.plot_widget)
        #self.x_axis = self.plot_widget.getAxis('bottom')  
          
        # 绘图所用的数据保存在这个变量中
        self.plot_data = {'time':deque(),'data':deque()}
        # 用一个定时器线程，每隔一定时长调用一次绘图函数 plot_once_event()，实现动态绘图
        self.plot_timer = QTimer(self)  
        self.plot_timer.timeout.connect(self.plot_once_event)


        # 连接数据库
        self.database = Database()


        ## ui文件中相关控件连接槽函数
        # 录入数据和绘图的按钮，连接对应函数
        self.action_insert.triggered.connect(self.insert_database)
        self.action_plot.triggered.connect(self.select_plot)

        # 控制绘图速度和长度的两个控件，如果值改变了就调用对应函数更新变量
        self.box_speed.valueChanged.connect(self.speed_change)
        self.box_length.valueChanged.connect(self.length_change)
        # 绘图速度和长度的两个变量，初始化时要读取下ui文件中设置的初始值
        self.plot_speed = max(int(self.box_speed.value()/5), 1)
        self.plot_length = max(int(self.box_length.value()*10), 1)


        # 更新树形表（查询数据库，查看存在哪些csv文件，将它们显示在GUI左侧的树形表中）
        self.update_tree_database()
 

    def insert_database(self):
        csv_path = QFileDialog.getOpenFileName(self,'选择文件','','csv files(*.csv)')[0]
        #csv_path = u'data_OK/TestData/甩负荷工况/甩负荷-1000s_0.1_0618.csv'
        basename = os.path.basename(csv_path)
        [table_name,filename] = basename.split('-')
        #print(table_name,filename)
        filename = '\''+filename+'\''
        if self.database.check_file_exist(table_name,filename):
            msg_box = QMessageBox(QMessageBox.Warning, 'Warning', '已经录入过此文件，禁止重复录入！')
            msg_box.exec_()
            return

        lines = pd.read_csv(csv_path).values.tolist()
        lines = [line+[filename] for line in lines]

        # 一次性不能插入太多条数据，否则sql会崩溃
        steps = len(lines)//3000
        for step in range(steps):
            self.database.insert_lines(table_name, lines[step*3000:(step+1)*3000])
        self.database.insert_lines(table_name, lines[steps*3000:])
    

        self.update_tree_database(table_name)



    # 绘图函数
    def select_plot(self):
        ## 确定当前树形表中选中的是哪个数据库
        this_item = self.tree_database.currentItem()
        if not this_item:   # 如果未被选中，则警告并退出
            msg_box = QMessageBox(QMessageBox.Warning, 'Warning', '请选择文件！')
            msg_box.exec_()
            return
        # 正确情况下，选中的应该是二级目录。因此如果当前选中的item没有父级目录，那么此item属于一级目录，警告并退出。
        parent = this_item.parent()
        if not parent:
            msg_box = QMessageBox(QMessageBox.Warning, 'Warning', '请选择文件！')
            msg_box.exec_()
            return
        # 确认无误，读取信息
        table_name = parent.text(0)
        file_name = this_item.text(0)


        ## 确定选中的参数类型
        this_item = self.tree_param.currentItem()
        if not this_item:
            msg_box = QMessageBox(QMessageBox.Warning, 'Warning', '请选择参数！')
            msg_box.exec_()
            return
        # 将中文转变为英文（举例：选中的是 “电功率”，那么将其转变为“diangonglv"，和数据库中统一
        key = ''.join(pinyin.get_pinyin(this_item.text(0)).split('-'))
        try:
            # 查询数据库
            ret = self.database.select(table_name,key,filename=file_name)
            # 开启绘图函数，并将数据赋给对应变量
            self.startPlotTimer(ret)
            # 更新文本信息
            self.label_data_name.setText(this_item.text(0))
        except:
            msg_box = QMessageBox(QMessageBox.Warning, 'Warning', '读取数据库失败！请检查数据库中是否存在该文件')
            msg_box.exec_()


    # 更新树形表（查询数据库，查看存在哪些csv文件，将它们显示在GUI左侧的树形表中）
    def update_tree_database(self,key=None):
        for i in range(self.tree_database.topLevelItemCount()):
            table_item = self.tree_database.topLevelItem(i)
            if key:
                if table_item.text(0) != key:
                    continue
            # 查看每个表中保存过的csv是否在tree里
            filename_saved = []
            for file_item in table_item.takeChildren():
                filename_saved.append(file_item.text(0))
            table_name = table_item.text(0)
            ret = self.database.select(table_name,'filename',time=False).tolist()
            hash = set([name[0] for name in ret])
            for file in list(hash):
                print(file)
                child=QTreeWidgetItem()
                child.setText(0,file)
                table_item.insertChild(0,child)


    def speed_change(self):
        self.plot_speed = max(int(self.box_speed.value()/5), 1)
    def length_change(self):
        self.plot_length = max(int(self.box_length.value()*10), 1)


    # 动态绘图函数，每次调用都会更新数据并绘图
    def plot_once_event(self):
        if len(self.plot_data['time'])==0 or self.plot_index>=len(self.plot_data['time'])-1:
            return

        # 更新条数（控制速度）      其实self.plot_data中包含了所有数据，但只需要绘制其中一部分。用一个序号来控制数据右端，序号递增实现数据的动态更新
        self.plot_index = min(self.plot_index+self.plot_speed , len(self.plot_data['time'])-1)  # 防止索引序号超出数据长度

        # 控制长度
        left_index = max(0,self.plot_index-self.plot_length)
        time,data = self.plot_data['time'][left_index:self.plot_index],self.plot_data['data'][left_index:self.plot_index]
 
        # 绘图
        self.plot_obj.setData(x=time, y=data)

        # 更新文本信息
        self.text_time.setText(str(round(self.plot_data['time'][self.plot_index],10))+' 秒')
        self.text_data.setText(str(round(self.plot_data['data'][self.plot_index],10)))

 

    # 绘图启动函数（开启绘图定时器，并加载数据、设置Y轴范围）
    def startPlotTimer(self,data):
        self.plot_index = 0
        self.plot_data['time'],self.plot_data['data'] = list(map(float,data[:,0])),list(map(float,data[:,1]))
        self.plot_widget.setYRange(max(0,min(self.plot_data['data'])), max(self.plot_data['data']))  
        self.plot_timer.start(50)#每隔50ms执行一次绘图函数 showTime

    # 中止绘图线程，并清空数据（实际上没有用到）
    def endPlotTimer(self):
        self.plot_timer.stop()#计时停止
        self.plot_data = {'time':deque(),'data':deque()}


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    stylesheet = qdarkstyle.load_stylesheet_pyqt5()
    app.setFont(QFont("微软雅黑", 9))
    app.setWindowIcon(QIcon("icon.ico"))
    #app.setStyleSheet(stylesheet)
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())
