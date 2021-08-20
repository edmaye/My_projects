import pymysql,json


def loadJson(jsonPath):
    with open(jsonPath, 'r',encoding='UTF-8') as f:
        data = json.load(f)
        return data
        
class Database():
    def __init__(self):
        self.loadConfig()
        self.connect()
    def connect(self):
        print(self.username,self.ip,self.password,self.database,self.port)
        self.conn = pymysql.connect(host=self.ip,
                                user=self.username,
                                password=self.password,
                                database=self.database,
                                port=self.port)
        self.cursor = self.conn.cursor()
        print('成功啦！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！')

        
    def loadConfig(self,name='softwareConfig'):
        result = loadJson(name+'.json')
        self.ip = result["ip"]
        self.username = result["username"]
        self.port = int(result["port"])
        self.password = result["password"]
        self.database = result["database"]
        self.yuangong = result['yuangong']
        self.zhiyeshangwang = result['zhiyeshangwang']
        self.zhiyeweihai = result['zhiyeweihai']
        self.ch_yuangong = result['ch_yuangong']
        self.ch_zhiyeshangwang = result['ch_zhiyeshangwang']
        self.ch_zhiyeweihai = result['ch_zhiyeweihai']
        # self.keys_name = []
        # self.keys_type = {} # 储存对应key的数据类型
        # for data in self.key_show:
        #     self.keys_name.append(data[0])
        #     self.keys_type[data[0]] = data[1]   

