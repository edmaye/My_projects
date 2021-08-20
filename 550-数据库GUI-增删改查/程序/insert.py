from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import numpy as np
import inspect, re

def varname(p):
	for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
		m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
	if m:
		return m.group(1)

class Insert(QMainWindow):
    def __init__(self,object):
        super(Insert, self).__init__()  # 调用父类的构造函数
        uic.loadUi(object+"xinxitianjia.ui", self)
        print(object+"xinxitianjia.ui")
        self.object = object
        self.pushButton_exit.clicked.connect(self.out)
    def out(self):
        exit()
    def getInfo(self):
        if self.object=='yuangong':
            name = self.lineEdit_name.text()
            id = self.lineEdit_id.text()
            telephone = self.lineEdit_telephone.text()
            zhicheng = self.lineEdit_zhicheng.text()
            zhiwu = self.lineEdit_zhiwu.text()
            bumen = self.lineEdit_bumen.text()
            address = self.lineEdit_address.text()
            sex = self.comboBox_sex.currentText()
            mianmao = self.comboBox_mianmao.currentText()
            data=self.dateEdit_birthday.date().toString(Qt.ISODate)
            keys = ['id,','xingming,','xingbie,','chushengriqi,','zhengzhimianmao,','jiatingdizhi,','lianxidianhua,','bumen,','zhichen,','zhiwu,']
            values_temp = [id,name,sex,data,mianmao,address,telephone,bumen,zhicheng,zhiwu]
            values = ['\''+value+"\'," for value in values_temp]
            base = "INSERT INTO " + 'yuangong'
        if self.object=='zhiyeshangwang':
            name = self.lineEdit_name.text()
            id = self.lineEdit_id.text()
            leixing = self.lineEdit_leixing.text()
            yuanyin = self.lineEdit_yuanyin.text()
            date=self.dateEdit_date.date().toString(Qt.ISODate)
            keys = ['id,','zhigongxingming,','shiguriqi,','shiguleixing,','shiguyuanyin,']
            values_temp = [id,name,date,leixing,yuanyin]
            values = ['\''+value+"\'," for value in values_temp]
            base = "INSERT INTO " + 'zhiyeshangwang'
        if self.object=='zhiyeweihai':
            name = self.lineEdit_name.text()
            id = self.lineEdit_id.text()
            telephone = self.lineEdit_telephone.text()
            zhicheng = self.lineEdit_zhicheng.text()
            zhiwu = self.lineEdit_zhiwu.text()
            bumen = self.lineEdit_bumen.text()
            address = self.lineEdit_address.text()
            sex = self.comboBox_sex.currentText()
            mianmao = self.comboBox_mianmao.currentText()
            data=self.dateEdit_birthday.date().toString(Qt.ISODate)
            keys = ['id,','zhiyeshangwangmingchen,','weixianjibie,','guanxiabumen,','shangwangriqi,','kongzhiyaoqiu','zerenren','jianchariqi']
            values_temp = [id,name,sex,data,mianmao,address,telephone,bumen,zhicheng,zhiwu]
            values = ['\''+value+"\'," for value in values_temp]
            base = "INSERT INTO " + 'zhiyeweihai'
        keys_str = "("
        values_str = " values ("
        if (len(keys) > 0):
            for key,value in zip(keys,values):
                if value=='\'\',':
                    continue
                keys_str += key
                values_str += value
            keys_str = keys_str[:-1]+')'
            values_str = values_str[:-1]+')'
        else:
            return None,None
        base = base + keys_str + values_str + ';'
        return base

import sys
if __name__=='__main__':
    app = QApplication(sys.argv)

    app.setFont(QFont("微软雅黑", 9))
    gui = Insert('yuangong')
    gui.show()
    sys.exit(app.exec_())
