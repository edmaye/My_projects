import pymysql,json
import numpy as np

def loadJson(jsonPath):
    with open(jsonPath, 'r',encoding='UTF-8') as f:
        data = json.load(f)
        return data
        
class Database():
    def __init__(self):
        # 加载配置文件中的信息
        self.loadConfig()
        
        # 连接数据库
        self.conn = pymysql.connect(host=self.ip,
                                user=self.username,
                                password=self.password,
                                database=self.database,
                                port=self.port)
        self.cursor = self.conn.cursor()

        
    def loadConfig(self,name='config'):
        result = loadJson(name+'.json')
        self.ip = result["ip"]
        self.username = result["username"]
        self.port = int(result["port"])
        self.password = result["password"]
        self.database = result["database"]
        self.keys = result["keys"]

    # 检查某个文件是否已经录入进数据库
    def check_file_exist(self,table_name,filename):
        sql = 'select filename from '+table_name+' where filename='+filename
        self.cursor.execute(sql)
        ret = np.array(self.cursor.fetchall())
        if len(ret)>0:
            print('已经录入过此文件，跳过')
            return True
        return False

    # 插入数据库
    def insert_lines(self,table_name,lines):
        # sql命令
        sql = 'insert into '+table_name+' (time,diangonglv,fanyingduigonglv,qilunjigonglv,yasuojigonglv,fanyingduirukouwendu,fanyingduichukouwendu,qilunjichukouwendu,yasuojirukouwendu,yasuojichukouwendu,liuliang,zhuansu,filename) values'
        for values in lines:
            sent = '('
            for value in values:
                if not isinstance(value,str):
                    value = str(value)
                sent += value+','
            sent = sent[:-1]+'),'
            sql += sent
        sql = sql[:-1]
        # 生成的sql语句
        self.cursor.execute(sql)
        self.conn.commit()

    # 查询数据库
    def select(self,table_name,key,time=True,filename=None):
        sql = 'select '
        if time:
            sql += 'time,'
        sql += key +' from '+table_name
        if filename:
            sql += ' where filename='+'\''+filename+'\''
        print(sql)
        self.cursor.execute(sql)
        ret = np.array(self.cursor.fetchall())
        return ret
    
    # 供测试用的函数，不用管
    def test(self):
        sql = 'select * from 热阱'
        self.cursor.execute(sql)
        ret = np.array(self.cursor.fetchall())
        print(ret)


if __name__=='__main__':
    database = Database()
    database.test()