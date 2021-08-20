import numpy as np  
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.tree import DecisionTreeRegressor  # 决策树
from sklearn.decomposition import PCA   # PCA用于将图像进行降维度
from sklearn.metrics import  mean_squared_error # 计算MSE指标，用于评估模型精度
import cv2,time

def ml_process(kind,Model):
    """
        模型训练+评估
    """
    # 读取txt中的数据
    img_path = []
    label = []
    with open('data_'+kind+'.txt', mode='r') as f:
        lines = f.readlines()
        for i in range(1,len(lines)):
            line = lines[i].strip().split(',')
            img_path.append(line[0])
            label.append(list(map(float,line[1:])))
    label = np.array(label)

    # 加载图像并reshape成一维向量（供PCA降维）
    data = np.zeros((len(img_path),600*450*3),dtype=np.uint8)
    for i,path in enumerate(img_path):
        img = cv2.imread(path)
        img = cv2.resize(img,(600,450))
        img = img.reshape(1,600*450*3)
        data[i,:] = img

    # 特征降维（PCA提取特征）
    n_components = len(img_path)   # 降维后的特征维数（不能大于样本数，因此样本数太少会有影响）
    data_PCA = PCA(n_components=n_components, svd_solver='auto', whiten=True).fit(data)
    data_pca = data_PCA.transform(data)      


    # 划分训练集/测试集
    split_index = int(data_pca.shape[0]*0.8)    # 划分界限，80%数据处
    x_train = data_pca[:split_index]
    y_train = {'trust':label[:split_index,0], 'read':label[:split_index,1]}
    x_test = data_pca[split_index:]
    y_test = {'trust':label[split_index:,0], 'read':label[split_index:,1]}

    # 针对trust数据进行训练，并计算精度
    model_trust = Model()   
    model_read = Model() 

    # 模型训练
    t1 = time.time()
    model_trust.fit(x_train, y_train['trust'])  
    model_read.fit(x_train, y_train['read'])
    t2 = time.time()

    # 模型预测
    predict_trust = model_trust.predict(x_test) 
    predict_read = model_read.predict(x_test)
    t3 = time.time()
    
    # 模型评估，计算测试集精度
    mse_trust = mean_squared_error(predict_trust,y_test['trust'])   # MSE指标
    mse_read = mean_squared_error(predict_read,y_test['read'])
    
    time_train = t2-t1
    time_predict = t3-t2
    return mse_trust, mse_read, time_train, time_predict


if __name__=='__main__':

    for kind in ['PB','PD','PP']:
        mse_trust,mse_read,time_train,time_predict = ml_process(kind,SVR)
        print('SVM-%s  trust：%f  read：%f  训练耗时: %fs  预测耗时：%fs'%(kind,round(mse_trust,5),round(mse_read,5),round(time_train,7),round(time_predict,7)))


    for kind in ['PB','PD','PP']:
        mse_trust,mse_read,time_train,time_predict = ml_process(kind,DecisionTreeRegressor)
        print('决策树-%s  trust：%f  read：%f  训练耗时: %fs  预测耗时：%fs'%(kind,round(mse_trust,5),round(mse_read,5),round(time_train,7),round(time_predict,7)))




