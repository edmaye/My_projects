import cv2
import numpy as np

src = cv2.imread('3_json/label.png')

color = set()
ccc = []
for i in range(src.shape[0]):
    for j in range(src.shape[1]):
        #print(src.shape,i,j)
        a,b,c = src[i,j,:]
        cc = str(a)+'-'+str(b)+'-'+str(c)
        if cc not in color:
            color.add(cc)
            ccc.append(src[i,j,:])
print(ccc)
