import numpy as np
from PIL import Image, ImageChops, ImageFilter
import time
print('>>>> PIL 前背景分离 <<<<')
# 加载背景和目标图像
t = time.time()
bg = Image.open('background.png')  
target = Image.open('target.png')  
print('加载图片耗时：',1000*(time.time()-t),'毫秒')

# 图像减背景，消除背景,得到前景掩模
t = time.time()
mask = Image.blend(target,ImageChops.invert(bg) ,0.5)
mask = mask.convert('L')
mask = mask.point(lambda i: i < 126 and 1)
print('提取前景掩模耗时：',1000*(time.time()-t),'毫秒')

# 中值滤波去除小噪点
t = time.time()
mask = mask.filter(ImageFilter.MedianFilter)
print('滤波去噪耗时：',1000*(time.time()-t),'毫秒')

# 掩模运算，将利用前景掩模提取原图对应区域，将前景提取出来
t = time.time()
arr = np.array(target)  # 将图像转为数组
arr = np.transpose(arr, [2, 0, 1])  # array维度 [W, H, C] -> [C, W, H]，这样才支持点乘操作
arr = arr * mask    # 点乘
arr = np.transpose(arr, [1, 2, 0])  # array维度 [C, W, H] -> [W, H, C]，转回原图形式
front = Image.fromarray(arr, mode='RGB') # 将数组转回图像格式
print('掩模运算耗时',1000*(time.time()-t),'毫秒')

# 保存图片
t = time.time()
front.save('front_PIL.png')
print('保存图片耗时：',1000*(time.time()-t),'毫秒')

# 可视化
front.show()
