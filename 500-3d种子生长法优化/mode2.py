import numpy as np
from PIL import Image
import cv2,time,os
import numba as nb

#@nb.jit()
def regionGrowing(grayImg, seed, threshold):
    """
    :param grayImg: 灰度图像
    :param seed: 生长起始点的位置
    :param threshold: 阈值
    :return: 取值为{0, 255}的二值图像
    """
    [maxX, maxY, maxZ] = grayImg.shape[0:3]

    # 用于保存生长点的队列
    pointQueue = []
    pointQueue.append((seed[0], seed[1],seed[2]))


    checked = np.zeros_like(grayImg,dtype=bool) # 是否被检验过



    pointsNum = 1
    pointsMean = float(grayImg[seed[0], seed[1], seed[2]])
    print('初始值:',pointsMean,' 阈值:',threshold)
    # 用于计算生长点周围26个点的位置
    Next26 = [[-1, -1, -1], [-1, 0, -1], [-1, 1, -1],
                [-1, 1, 0], [-1, -1, 0], [-1, -1, 1],
                [-1, 0, 1], [-1, 0, 0], [-1, 0, -1],
                [0, -1, -1], [0, 0, -1], [0, 1, -1],
                [0, 1, 0], [-1, 0, -1],
                [0, -1, 0], [0, -1, 1], [-1, 0, -1],
                [0, 0, 1], [1, 1, 1], [1, 1, -1],
                [1, 1, 0], [1, 0, 1], [1, 0, -1],
                [1, -1, 0], [1, 0, 0], [1, -1, -1]]

    while(len(pointQueue)>0):
        growSeed = pointQueue[0]
        del pointQueue[0]

        # 是否被检验过
        if(checked[growSeed[0],growSeed[1],growSeed[2]] == True):
            continue
        checked[growSeed[0],growSeed[1],growSeed[2]] = True



        # 符合条件则生长，并将邻域点加入到生长点队列中
        data = grayImg[growSeed[0],growSeed[1],growSeed[2]]
        
        if(abs(data - pointsMean)<threshold):

            pointsNum += 1
            pointsMean = (pointsMean * (pointsNum - 1) + data) / pointsNum
            # 添加邻域点
            for differ in Next26:
                growPointx = growSeed[0] + differ[0]
                growPointy = growSeed[1] + differ[1]
                growPointz = growSeed[2] + differ[2]
                
                # 是否是边缘点
                if((growPointx < 0) or (growPointx >= maxX) or
                    (growPointy < 0) or (growPointy >= maxY) or (growPointz < 0) or (growPointz >= maxZ)) :
                    continue
                pointQueue.append([growPointx, growPointy, growPointz])

        # 返回的不是矩阵，而是pointsMean这个阈值参数
        if pointsNum>10000:
            return pointsMean
    return pointsMean



seed = [0,0,0]
seed_ok = False

def capture_event(event,x,y,flags,params):
    global seed
    global seed_ok
    if event==cv2.EVENT_LBUTTONDOWN:
        #cv2.circle(image,(x,y),30,(0,0,255),-1)
        seed = [y,x,seed[2]]
        print('初始点：',seed)
        seed_ok = True

 

if __name__ == '__main__':
    if not os.path.exists('outimage'):
        os.makedirs('outimage')
    # >>>>>>>>>>>>>>>>>>>>>>>>  鼠标点击选取初始点  <<<<<<<<<<<<<<<<<<<<<<<<<<
    index = 0   # 从第几张图像上选取初始点
    seed[2] = index
    image=cv2.imread('./use/XY_'+str(index)+'.jpg') # 加载对应图片
    ## 下面这段不用改
    window = 'choose seed'
    cv2.namedWindow(window)
    cv2.setMouseCallback(window,capture_event)
    while True:
        cv2.imshow(window,image)
        key = cv2.waitKey(1)
        if key==13 or key==27 or seed_ok == True:
            break
    cv2.destroyAllWindows()


    # >>>>>>>>>>>>>>>>>>>>>>>>  加载图像  <<<<<<<<<<<<<<<<<<<<<<<<<<
    size = 1132     # 不resize
    imgall = np.zeros((size, size, 400))      # 预先分配好大矩阵，然后直接赋值。这样效率高得多
    t1 = time.time()
    for i in range(400):
        name = f"./use/XY_{i}.jpg"
        im = cv2.imread(name)
        img = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        if size!=1132:
            img = cv2.resize(img,(size,size))
        imgall[:,:,i] = img   
    t2 = time.time()
    print('图像加载完毕，耗时',t2-t1,'s')

    
    # >>>>>>>>>>>>>>>>>>>>>>>>  区域生长法分割  <<<<<<<<<<<<<<<<<<<<<<<<<<
    mean_threshold = regionGrowing(imgall, [round(seed[0]/1132.0*size), round(seed[1]/1132.0*size), seed[2]], 20)
    t3 = time.time()
    print('种子生长完毕，耗时',t3-t2,'s')


    # >>>>>>>>>>>>>>>>>>>>>>>>  保存图像  <<<<<<<<<<<<<<<<<<<<<<<<<<
    print('二值化阈值区间中心:',mean_threshold)
    for j in range(400):
        img_tmp = cv2.inRange(imgall[:, :, j],mean_threshold-20,mean_threshold+20)     # 提取灰度在指定区间内的像素（区间二值化）
        cv2.imwrite(f'./outimage/my{j}.jpg',img_tmp)
    t4 = time.time()
    print('图像保存完毕，耗时',t4-t2,'s')