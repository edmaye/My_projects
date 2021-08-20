# HOG
import cv2
import numpy as np
# 判断矩形i是否完全包含在矩形o中
def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih

# 绘制颜色框
def draw_person(image, person):
    x, y, w, h = person
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

    img = cv2.imread("12.jpg")
def xingzhuang(img):
    hog = cv2.HOGDescriptor()  # 启动检测器对象
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  # 指定检测器类型为人体
    found, w = hog.detectMultiScale(img, 0.1, (1, 1))    # 加载并检测图像
    print(found)
    print(w)

    # 丢弃某些完全被其它矩形包含在内的矩形
    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and is_inside(r, q):
                break
        else:
            found_filtered.append(r)
            print(found_filtered)
    # 对不包含在内的有效矩形进行颜色框定
    for person in found_filtered:
        draw_person(img, person)
    return img



if __name__=='__main__':
    img = cv2.imread('12.jpg')
    th = xingzhuang(img)
    cv2.imshow('split', th)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
