from cnn_main import cnn_process
from ml_main import ml_process
from sklearn.svm import SVR  # SVM中的回归算法
from sklearn.tree import DecisionTreeRegressor  # 决策树
import warnings,torch
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning)

# 固定随机种子
seed = 2021
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

"""
    依次调用三种算法、对三种类型数据进行处理
    ml_main.py 中的 ml_process 函数：   
    cnn_main.py中的 cnn_process 函数：
"""
# 依次对三种类型的数据进行处理
for kind in ['PB','PD','PP']:
    # SVM
    mse_trust, mse_read, time_train, time_predict = ml_process(kind,SVR)
    print('SVM   -%s  MSE-trust：%f  MSE-read：%f  训练耗时: %fs   预测耗时：%fs'%(kind,round(mse_trust,5),round(mse_read,5),round(time_train,7),round(time_predict,7)))
    # 决策树
    mse_trust, mse_read, time_train, time_predict = ml_process(kind,DecisionTreeRegressor)
    print('决策树-%s  MSE-trust：%f  MSE-read：%f  训练耗时: %fs   预测耗时：%fs'%(kind,round(mse_trust,5),round(mse_read,5),round(time_train,7),round(time_predict,7)))
    # CNN
    mse_trust, mse_read, time_train, time_predict = cnn_process(kind)
    print('CNN   -%s  MSE-trust：%f  MSE-read：%f  训练耗时: %fs  预测耗时：%fs'%(kind,round(mse_trust,5),round(mse_read,5),round(time_train,7),round(time_predict,7)))

