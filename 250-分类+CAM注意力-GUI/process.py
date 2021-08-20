#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-21 14:16:48
# @Author  : Lewis Tian (chtian@hust.edu.cn)
# @Link    : https://lewistian.github.io
# @Version : Python3.7

import time,os,sys,cv2,json
import numpy as np
import matplotlib.pyplot as plt
#import QMutex
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
# model
import torch
from torch import nn
from torchvision import transforms
from skimage import io
from torch._C import device
from torchvision import models
from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from interpretability.guided_back_propagation import GuidedBackPropagation
from models.mobilenetv2 import MobileNetV2



class_txt = {
    0:'第1类：少时诵诗书所所所',
    1:'第2类：不知道发if哈卡算法耗时',
    2: ("第3类：zzzzzzzzzzzzzzzzzzzzzzzzzzzzzz\n"
        "第二行第二行第二行第二行第二行第二行第二行"),
        
    3: ("第4类：xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n "
        "xxxxxxxxxxxxxxxxxxx少时诵诗书所所所少时\n"
        "诵诗书所所所少时诵诗书所所所少时诵诗书所所\n"
        "时诵诗书所所所少时诵诗书所所所少时诵诗书所所\n"
        "所少时诵诗书所所所"),
    4:'第5类：cccccccc222222222222222cccccccccccccccccccccccccccccccccccccccc少时诵诗书所所所少时诵诗书所所所少时诵诗书所所所少时诵诗书所所所少时诵诗书所所所少时诵诗书所所所',
    5:'第6类：vvvvvvvv8888888888888vvvvvvvvv少时诵诗书所',
    6:'第7类：nnnnnnnnnnnnn=================nnnnnnnnnnnnnnnnnn少时诵诗书所所所',
    7:'第8类：三生三世付付无若过无过无',
}







def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name

def prepare_input(image):
    image = image.copy()

    # 归一化
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # channel first
    image = image[np.newaxis, ...]  # 增加batch维

    return torch.tensor(image, requires_grad=True)

def gen_gb(grad):
    """
    生guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb
def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)
def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    
    return norm_image(cam), (heatmap * 255).astype(np.uint8)

class ProcessThread(QThread):
    # 视频处理线程
    result = pyqtSignal(dict)  # 一副图像完成处理后，以信号量的方式传出去
    result_table = pyqtSignal(list)
    def __init__(self):
        super(ProcessThread,self).__init__()
        self.running = False
        self.stop_flag =False

        self.model_init()
    def model_init(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = MobileNetV2(num_classes=8)
        self.model.load_state_dict(torch.load("./model-epoch-82-acc-0.977728285077951.pth", map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.data_transform = transforms.Compose(
                [transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        self.class_indict = json.load(json_file)

    def is_running(self):
        return self.running

    def set_imgpath(self,img_path):
        self.img_path = img_path

    def run(self):
        self.running = True
        src = Image.open(self.img_path).convert("RGB")
        src_array = np.array(src)
        # >>>>>>>>>>>>>>   GUI显示-原图 <<<<<<<<<<<<<<<<<<<<<
        img_tmp = cv2.resize(src_array,(500,400))
        self.result.emit({'img':img_tmp,'label':'label_origin'})
        img = self.data_transform(src).to(self.device)
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            # predict class
            output = torch.squeeze(self.model(img.to(self.device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            
            self.result_table.emit(list(predict.numpy()))
    
            self.result.emit({'log':{'txt':class_txt[int(predict_cla)],'label':'label_classtxt'}})

        ## 热力图
        image_dict = {}
        img = np.float32(cv2.resize(src_array, (224, 224))) / 255
        inputs = prepare_input(img)
        net = models.resnet50(pretrained=True)
        layer_name = get_last_conv_name(net)
        grad_cam = GradCAM(net, layer_name)
        mask = grad_cam(inputs, None)  # cam mask
        image_dict['cam'], image_dict['heatmap'] = gen_cam(img, mask)

        # >>>>>>>>>>>>>>   GUI显示-CAM热力图 <<<<<<<<<<<<<<<<<<<<<
        print_r = "class: {}   prob: {:.3}".format(self.class_indict[str(predict_cla)], predict[predict_cla].numpy())
        img_tmp = cv2.resize(np.array(image_dict['cam']),(256,256))
        self.result.emit({'img':img_tmp, 'label':'label_heatmap1', 'log':{'txt':print_r,'label':'label_heatmap1_txt'}})
        

        # Grad-CAM++
        grad_cam.remove_handlers()
        grad_cam_plus_plus = GradCamPlusPlus(net, layer_name)
        mask_plus_plus = grad_cam_plus_plus(inputs, None)  # cam mask
        image_dict['cam++'], image_dict['heatmap++'] = gen_cam(img, mask_plus_plus)

        # >>>>>>>>>>>>>>   GUI显示-CAM++热力图 <<<<<<<<<<<<<<<<<<<<<
        print_r = "class: {}   prob: {:.3}".format(self.class_indict[str(predict_cla)], predict[predict_cla].numpy())
        img_tmp = cv2.resize(np.array(image_dict['cam']),(256,256))
        self.result.emit({'img':img_tmp, 'label':'label_heatmap2', 'log':{'txt':print_r,'label':'label_heatmap2_txt'}})


        ## 柱状图
        plt.hist(cv2.cvtColor(src_array,cv2.COLOR_BGR2GRAY).ravel(), 256, [0, 256])
        plt.savefig('temp.png')
        plt.close('all')
        bar = cv2.imread('temp.png')
        img_tmp = cv2.resize(bar,(512,320))
        self.result.emit({'img':img_tmp, 'label':'label_bar'})
        os.remove('temp.png')
        self.running = False

  
