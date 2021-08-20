# 三种模型结构说明

- MLP（测试集均方根损失：0.00233）

  网络结构

  ```python
  Linear(10,30),   
  ReLU(),   
  Linear(30,15), 
  ReLU(),   
  Linear(15,10)  
  ```

  loss：均方根损失

- GAN_no_noise（测试集均方根损失：0.00238）

  生成器G（和上文所用的MLP结构一致）

  ```python
  Linear(10,30),   
  ReLU(),   
  Linear(30,15), 
  ReLU(),   
  Linear(15,10)
  ```

  loss：均方根损失 + 对抗损失

  判别器D

  ```python
  Linear(10,6),   
  ReLU(),   
  Linear(6,3), 
  ReLU(),   
  Linear(3,1)
  Sigmoid()
  ```

  loss：交叉熵损失

- GAN_with_noise（测试集均方根损失：0.00066）   

  - 说明

    上一个GAN，说穿了就是MLP加了个**对抗损失**（利用鉴别器D来实现，训练过程和传统GAN一致）。但GAN原本是用来做生成任务的，输入只有噪声z。其核心除了**对抗损失**之外，还有学习噪声分布到目标域分布之间的变换关系。

    之后出现了conditional GAN，生成器G的输入不再只有噪声z，还有标签y，以约束G的输出减少随机性。      

    本文仿照conditional GAN的结构，G的输入除了特征向量x之外还有随机噪声z，因此第一层全连接层的输入维度从10变为20.

  生成器G（**仿照conditional GAN**，输入除了1x10特征外，还cat了一个1x10噪声）

  ```python
  Linear(10+10,30),   
  ReLU(),   
  Linear(30,15), 
  ReLU(),   
  Linear(15,10)
  ```

  loss：均方根损失 + 对抗损失

  判别器D

  ```python
  Linear(10,6),   
  ReLU(),   
  Linear(6,3), 
  ReLU(),   
  Linear(3,1)
  Sigmoid()
  ```

  loss：交叉熵损失