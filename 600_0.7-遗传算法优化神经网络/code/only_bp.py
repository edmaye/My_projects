import os,glob
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import _base
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix


from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# file_list = glob.glob("./face/rawdata/*")                        #file_list为遍历图像文件名称 list类型
# rows = 128   #行数
# cols = 128   #列数
# channal = 1  #通道数，灰度图为1
# imgdata = np.empty((1,16384),dtype='uint8') #创建空数组存放图像信息

# # 遍历图像信息存入imgdata中
# # imgdata为 3991*16384 维的数组（除去'2416'和'2412'（图像大小存在错误，暂不考虑））
# i = 0
# for file_name in file_list:
#     # i = i+1
#     # if i>1000:
#     #     break
#     img = np.fromfile(file_name, dtype='uint8')   #img为numpy.ndarray(数组)类型
#     img = img.reshape(1,16384)
#     print(file_name)
#     imgdata = np.vstack((imgdata,img))
# imgdata = np.delete(imgdata,0,axis=0)        #删除第一行；axis=0，为删除行；axis=1为删除列
# print("The size of imgs' characteristic matrix:")
# print(imgdata.shape)

# # 特征降维（PCA提取特征）
# n_components = 150   # 降维后的特征维数
# print("Extracting the top %d eigenfaces from %d faces" % (n_components, imgdata.shape[1]))
# imgdata_PCA = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(imgdata)
# eigenfaces = imgdata_PCA.components_.reshape((n_components,128,128))
# imgdata_pca = imgdata_PCA.transform(imgdata)       # imgdata_pca为降维后的图像信息，3991 * 200
# print("The size of PCA characteristic matrix:")
# print(imgdata_pca.shape)

# # 构造人脸图像库
# # 类别
# f = open(r"./face/face1.txt")
# lines = f.readlines()
# img_information = []
# imginfor = []
# wrong = []
# for line in lines:
#     temp1 = line.strip('\n')    # strip()为删除函数；删去空行
#     temp3 = temp1.split('(')    # 以 “（” 分割
#     img_information.append(temp3)
# for i in range(len(img_information)):
#     if img_information[i][1].strip() != '_missing descriptor)':
#         sex = img_information[i][1].strip()
#         gesture = img_information[i][4].strip()
#         age = img_information[i][2].strip()
#         if sex == '_sex  male)':
#             imginfor.append('male')
#         else:
#             imginfor.append('female')
#     else:
#         imginfor.append('missing')
#         wrong.append(i)
# # 暂时删除缺失图像信息的信息
# index_offset = 0          #索引的偏移量：因为下面是逐次删除，所以每当前一次删除后，后面的索引响应减一
# for i in range(len(wrong)):
#     a = wrong[i] - index_offset
#     del img_information[a]
#     del imginfor[a]
#     index_offset += 1
# offset = 0
# for j in wrong:
#     if j < 1189 and j > 0:
#         offset += 1
# del imginfor[1189-offset]          # 删除维度错误的图像  1189 smiling;1193 serious
# del imginfor[1193-offset-1]
# del img_information[1189-offset]
# del img_information[1193-offset-1]
# print("========================================")
# print("Total date:%d,\nmale:%d;\nfemale:%d."
#       % (len(imginfor),imginfor.count('male'),imginfor.count('female')))  #类别样本数 2425 1566
# print("========================================")
# f.close()

from data_generater import data_generate


data = data_generate("./face/rawdata/*","./face/face1.txt")
feature = data['feature']            # 降维后的特征
targets_names = data["label_name"]      # 类别
img_target = data["label"]


feature_name = []
for i in range(0,150):      # 特征编号
    feature_name.append(i)
face = _base.Bunch(data = feature,target = img_target,target_names = targets_names,feature_names = feature_name)  #face为人脸库

X_train, X_test, y_train, y_test = train_test_split(face.data,face.target, test_size=0.3 , random_state=2) # 划分训练集
print("The shape of training and of testing : ")
print(X_train.shape, X_test.shape)


########################################################################################################################
###  5、BP神经网络分类器分类
sc = StandardScaler()
sc.fit(X_train)
print(X_train.mean())
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
print(X_train_std.mean())
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5,2), random_state=1)

clf.fit(X_train_std, y_train)
print('每层网络层系数矩阵维度：\n',[coef.shape for coef in clf.coefs_])

#y_pred_BP = clf.predict(X_test_std)
#accrracy_BP = clf.score(y_pred_BP,y_test)
# print('预测结果：',y_pred_BP)
# y_pred_pro =clf.predict_proba(X_test_std)
# print('预测结果概率：\n',y_pred_pro)
print('The accuracy of train is :',clf.score(X_train, y_train))
print('The accuracy of test is :',clf.score(X_test, y_test))

cengindex = 0
# for wi in clf.coefs_:
#     cengindex += 1  # 表示底第几层神经网络。
#     print('第%d层网络层:' % cengindex)
#     print('权重矩阵维度:',wi.shape)
#     print('系数矩阵:\n',wi)




