import cv2
import time
print('>>>> opencv 前背景分离 <<<<')
# 加载背景和目标图像
t = time.time()
bg = cv2.imread('background.png')
target = cv2.imread('target.png')
print('加载图片耗时：',1000*(time.time()-t),'毫秒')

# 图像减背景，消除背景,得到前景掩模
t = time.time()
mask = cv2.subtract( bg,target, dst=None, mask=None, dtype=None)    # 目标图 - 背景图 = 前景图
mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)    # 转灰度
_,mask = cv2.threshold(mask,10,255,cv2.THRESH_BINARY)   # 二值化
print('提取前景掩模耗时：',1000*(time.time()-t),'毫秒')

# 中值滤波去除小噪点
t = time.time()
mask = cv2.medianBlur(mask, 5)
print('滤波去噪耗时：',1000*(time.time()-t),'毫秒')

# 前景掩模优化，填充连通域内部的孔洞
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area <= 80000:   cv_contours.append(contour)
cv2.fillPoly(mask, cv_contours, (255, 255, 255))

# 掩模运算，将利用前景掩模提取原图对应区域，将前景提取出来
t = time.time()
front = cv2.bitwise_and(target, target, mask=mask)  # 删除了ROI中的logo区域(mask的logo为黑色，故and后该区域被舍去)---》上图中的图一
print('掩模运算耗时：',1000*(time.time()-t),'毫秒')

# 保存图片
t = time.time()
cv2.imwrite('front_opencv.png',front)
print('保存图片耗时：',1000*(time.time()-t),'毫秒')

# 可视化，顺便保存下提取的背景
background = target-front
cv2.imwrite('background_opencv.png',background)
cv2.imshow('front_opencv',front)
cv2.imshow('background_opencv',background)
cv2.waitKey(0)