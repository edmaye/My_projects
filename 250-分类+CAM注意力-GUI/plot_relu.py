
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# %matplotlib inline
sns.set(style="darkgrid")

fig = plt.figure(figsize=(12,6))

plt.xlim([-2, 2]);
plt.ylim([-1, 1.6]);

# 定义数值
x = np.sort(np.linspace(-10,10,1000))

# ReLu 函数
relu= [max(item,0) for item in x]

# LeakReLu函数
alpha = 0.1
leakRelu = [item if item>0 else item*alpha for item in x]

# PReLu函数
alpha = 0.1 # 可以学习的参数
leakRelu = [item if item>0 else item*alpha for item in x]

# ELU函数
alpha = 0.2
elu = [item if item>0 else (np.exp(item)-1)*alpha for item in x]

# SELU函数
alpha = 1
r = 0.5
selu = [item if item>0 else (np.exp(item)-1)*alpha for item in x]
selu = list(map(lambda x:x*r,selu))

# 绘图
plt.plot(x,relu,color="#ff0000", label = r"ReLu", marker='*')
plt.plot(x,leakRelu,color="#0000ff", label = r"LeakReLu")
plt.plot(x,elu,color="#00ff00", label = r"ELU")
plt.plot(x,selu,color="#00ffee", label = r"SELU")

plt.legend(prop={'family' : 'Times New Roman', 'size'   : 16})
plt.show()