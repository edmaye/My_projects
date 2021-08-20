import cv2

def ostuyuzhi(img):
    # 先进行高斯滤波，再使用Otsu阈值法
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


if __name__=='__main__':
    img = cv2.imread('12.jpg', 0)
    th = ostuyuzhi(img)
    cv2.imshow('split', th)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

