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

    outImg = np.zeros_like(grayImg,dtype=bool)  # 是否为正
    checked = np.zeros_like(grayImg,dtype=bool) # 是否被检验过

    outImg[seed[0], seed[1], seed[2]] = True

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
            
            outImg[growSeed[0],growSeed[1],growSeed[2]] = True

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
        # if pointsNum>10000:
        #     return pointsMean
    return outImg



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
    size = 256  # 将图像从1132缩放到一个较小的尺寸（可以选择256/512/700等，只要缩放倍数 n 不要大于5，一般结果失真现象不会很严重，但计算时间能减少n平方)
                # 如果要最终分割更精细的话，请将size调大（甚至调回原图尺寸1132）

    imgall = np.zeros((size, size, 400))      # 预先分配好大矩阵，然后直接赋值。  不要一个矩阵一个矩阵合并，那样矩阵大了之后每次合并都会开辟一个巨大空间，耗时越来越长。
    t1 = time.time()
    for i in range(400):
        name = f"./use/XY_{i}.jpg"
        im = cv2.imread(name)
        img = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)   # BGR转灰度图，原来的代码没有这一步是不对的，因为图像本身是灰度图只不过以RGB形式存储，一张图应该只对应一层。
        if size!=1132:
            img = cv2.resize(img,(size,size))   # 缩放
        imgall[:,:,i] = img   
    t2 = time.time()
    print('图像加载完毕，耗时',t2-t1,'s')

    
    # >>>>>>>>>>>>>>>>>>>>>>>>  区域生长法分割  <<<<<<<<<<<<<<<<<<<<<<<<<<
    outImg = regionGrowing(imgall, [round(seed[0]/1132.0*size), round(seed[1]/1132.0*size), seed[2]], 20)   #这里不直接输入seed，是因为将原图进行了缩放以减少计算量，因此坐标也要对应地进行缩放
    if size!=1132:
        outImg = cv2.resize(outImg.astype(np.uint8),(1132,1132))
    t3 = time.time()
    print('种子生长完毕，耗时',t3-t2,'s')


    # >>>>>>>>>>>>>>>>>>>>>>>>  保存图像  <<<<<<<<<<<<<<<<<<<<<<<<<<
    for j in range(400):
        img_tmp = outImg[:, :, j]
        img_tmp = cv2.morphologyEx(img_tmp, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) # 图像闭运算，让分割图平滑一点
        #img_tmp = cv2.inRange(imgall[:, :, j],thresh-20,thresh+20)     # 直接阈值全图操作法，后续选择用不用
        #img_tmp = img_tmp*255  # j的步进为1，因此不能按上面那样写
        cv2.imwrite(f'./outimage/my{j}.jpg',img_tmp*255)
    t4 = time.time()
    print('图像保存完毕，耗时',t4-t2,'s')