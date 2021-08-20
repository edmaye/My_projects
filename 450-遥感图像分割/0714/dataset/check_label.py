import cv2
import numpy as np
import glob

paths = glob.glob('label/3_*')
path = paths[5]

mask_origin = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

print(mask_origin.min(),mask_origin.max())
mask = np.zeros((mask_origin.shape[0],mask_origin.shape[1],6), dtype=np.float)
for i in range(6):
    mask[:,:,i] = mask_origin==i
    cv2.imwrite(str(i)+'____.png',mask[:,:,i]*255)
exit()