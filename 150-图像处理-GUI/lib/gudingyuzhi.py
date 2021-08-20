import cv2

def gudingyuzhi(img):
    #固定阈值分割
    ret, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return th

if __name__=='__main__':
    img = cv2.imread('split.jpg', 0)
    th = gudingyuzhi(img)
    cv2.imshow('split', th)
    cv2.waitKey(0)
    cv2.destroyAllWindows()