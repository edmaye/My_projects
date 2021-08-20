# Canny边缘检测算法
import cv2




def bianyuan(img):
    # 先进行高斯滤波，再使用Otsu阈值法
    img = cv2.Canny(img, 80, 150, (3, 3))
    return img


if __name__=='__main__':
    img = cv2.imread('12.jpg', 0)
    th = bianyuan(img)
    cv2.imshow('split', th)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
