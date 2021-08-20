# 高斯加权自适应阈值
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('split.jpg', 0)

th = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
cv2.imshow('split', th)
cv2.waitKey(0)
cv2.destroyAllWindows()
