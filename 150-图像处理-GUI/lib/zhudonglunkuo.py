import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

Image = cv2.imread('12.jpg',1)
image = cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
img = np.array(image,dtype='float64')
plt.imshow(img,cmap = 'gray')
#print(img.shape)

IniLSF = np.ones([img.shape[0],img.shape[1]],img.dtype)
IniLSF[40:80,40:80]=-1
IniLSF = -IniLSF


def CV(LSF, img, nu, mu, epison, step):
    Drc = (epison / math.pi) / (epison * epison + LSF * LSF)
    #    Hea = 0.5*(1+(2/math.pi)*mat_math(LSF/epison,'atan'))
    Hea = 0.5 * (1 + (2 / math.pi) * np.arctan(LSF / epison))
    Iy, Ix = np.gradient(LSF)  ##q4#
    #    s = mat_math(Ix*Ix+Iy*Iy,"sqrt")
    s = np.sqrt(Ix * Ix + Iy * Iy)
    Nx = Ix / (s + 0.000001)
    Ny = Iy / (s + 0.000001)
    Mxx, Nxx = np.gradient(Nx)
    Nyy, Myy = np.gradient(Ny)
    cur = Nxx + Nyy

    Length = nu * Drc * cur

    Area = mu * Drc

    s1 = Hea * img
    s2 = (1 - Hea) * img
    s3 = 1 - Hea
    C1 = s1.sum() / Hea.sum()
    C2 = s2.sum() / s3.sum()

    CVterm = Drc * (-1 * (img - C1) * (img - C1) + 1 * (img - C2) * (img - C2))
    LSF = LSF + step * (Length + Area + CVterm)
    return LSF

nu = 0.0001*255*255
mu = 1
num = 10
epison = 1
step = 0.1
LSF = IniLSF
for i in range(1,num):
    LSF = CV(LSF,img,nu,mu,epison,step)
    if i%2 == 0:
        plt.contour(LSF,[0],linewidths = 3.0,linestyles = 'dotted',colors='r')
#    plt.draw()
plt.show()
print("over")
