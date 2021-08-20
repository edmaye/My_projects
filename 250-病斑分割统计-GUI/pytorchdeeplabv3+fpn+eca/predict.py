#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from deeplabv3 import uNet
from PIL import Image
import os

uNet = uNet()
imgs = os.listdir("./img")

for jpg in imgs:

    img = Image.open("./img/"+jpg)
    image = uNet.detect_image(img)
    image.save("./img_out/"+jpg)


# while True:
#     img = input('Input image filename:')
#     try:
#         image = Image.open(img)
#     except:
#         print('Open Error! Try again!')
#         continue
#     else:
#         r_image = pspnet.detect_image(image)
#         r_image.show()
