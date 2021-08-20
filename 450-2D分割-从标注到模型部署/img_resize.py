import os,glob
import cv2
 


if __name__ == "__main__":
    imgs = glob.glob('mydata/*.jpg')
    for path in imgs:
        name = os.path.basename(path)
        img = cv2.imread(path)
        img = cv2.resize(img,(640,480))
        cv2.imwrite(path,img)