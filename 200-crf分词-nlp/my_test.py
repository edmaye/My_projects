import sklearn_crfsuite


# DataProcess类，负责加载文本、转化成BIS型数据集
class DataProcess(object):
    def __init__(self,file_path):
        self.initialize(file_path)

    def q_to_b(self, q_str):
        """全角转半角"""
        b_str = ""
        for uchar in q_str:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif 65374 >= inside_code >= 65281:  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            b_str += chr(inside_code)
        return b_str

    def b_to_q(self, b_str):
        """半角转全角"""
        q_str = ""
        for uchar in b_str:
            inside_code = ord(uchar)
            if inside_code == 32:  # 半角空格直接转化
                inside_code = 12288
            elif 126 >= inside_code >= 32:  # 半角字符（除空格）根据关系转化
                inside_code += 65248
            q_str += chr(inside_code)
        return q_str

    def initialize(self,file_path):
        # 加载txt文本
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        words_list = [line.strip().split('  ') for line in lines if line.strip()]
        del lines

        """初始化字序列、标记序列 """
        self.word_seq = []
        self.tag_seq = []
        for sent in words_list:
            w,t = [u'<BOS>'],[]
            
            for word in sent:
                if len(word)==1:
                    w.append(word)
                    t.append('S')
                    continue
                for i,c in enumerate(word):
                    if i==0:
                        tag = 'B'
                    else:
                        tag = 'I'
                    w.append(c)
                    t.append(tag)
            if len(w)!= len(t):
                pass
            w += [u'<EOS>']
            self.word_seq.append(w)
            self.tag_seq.append(t)


    def extract_feature(self, word_grams):
        """特征选取"""
        features, feature_list = [], []
        for index in range(len(word_grams)):
            for i in range(len(word_grams[index])):
                word_gram = word_grams[index][i]
                feature = {u'w-1': word_gram[0], u'w': word_gram[1], u'w+1': word_gram[2],
                           u'w-1:w': word_gram[0] + word_gram[1], u'w:w+1': word_gram[1] + word_gram[2],
                           # u'p-1': self.pos_seq[index][i], u'p': self.pos_seq[index][i+1],
                           # u'p+1': self.pos_seq[index][i+2],
                           # u'p-1:p': self.pos_seq[index][i]+self.pos_seq[index][i+1],
                           # u'p:p+1': self.pos_seq[index][i+1]+self.pos_seq[index][i+2],
                           u'bias': 1.0}
                feature_list.append(feature)
            features.append(feature_list)
            feature_list = []
        return features

    def segment_by_window(self, words_list=None, window=3):
        """窗口切分"""
        words = []
        begin, end = 0, window
        for _ in range(1, len(words_list)):
            if end > len(words_list): break
            words.append(words_list[begin:end])
            begin = begin + 1
            end = end + 1
        return words

    def generator(self):
        """训练数据"""
        word_grams = [self.segment_by_window(word_list) for word_list in self.word_seq]
        features = self.extract_feature(word_grams)
        return features, self.tag_seq



# # CRF_MODEL，负责模型训练
class CRF_MODEL(object):
    def __init__(self,data_path):
        self.dataset = DataProcess(data_path)  # 加载文本并生成数据集
        self.model = None

    def train(self):
        # crf模型
        self.model = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.01,max_iterations=300, all_possible_transitions=True)
        # 加载数据
        x, y = self.dataset.generator()
        # 训练
        self.model.fit(x, y)

    def predict(self, sentence):
        """预测"""
        u_sent = self.dataset.q_to_b(sentence)
        word_lists = [[u'<BOS>'] + [c for c in u_sent] + [u'<EOS>']]
        word_grams = [self.dataset.segment_by_window(word_list) for word_list in word_lists]
        features = self.dataset.extract_feature(word_grams)
        # 模型预测，得到每个字符的标签
        y_predict = self.model.predict(features)[0]
        # 根据标签划分字符
        entity = u'' + u_sent[0]
        for i in range(1,len(y_predict)):
            tag = y_predict[i]
            if tag=='B' or tag=='S':
                entity += u'  '
            entity += u_sent[i]
        return entity



if __name__=='__main__':
    

    model = CRF_MODEL('pku_training.txt')
    model.train()

    result = []
    with open('pku_test.txt','r',encoding='utf-8') as f:
        test_data = f.readlines()
        for line in test_data:
            result.append(model.predict(line))
    with open('result.txt','w',encoding='utf-8') as f:
        f.writelines(result)

