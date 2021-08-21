from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import Cityscapes
from torchvision import transforms as T
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib.pyplot as plt
from glob import glob

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, default='test_images', required=True,
                        help="单张图片或文件夹的路径")

    # 选择网络结构（为deeplabv3/deeplabv3+ 提供各种backbone)
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenetv3',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet',
                                 'deeplabv3plus_mobilenetv3'])
    
    parser.add_argument("--ckpt", default='checkpoints/best_deeplabv3plus_mobilenetv3_cityscapes_os16.pth.tar', type=str,
                        help="对应的权重文件")
                        
    return parser

def main():
    opts = get_argparser().parse_args()
 

    opts.num_classes = 19
    decode_fn = Cityscapes.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.'+ext), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet,
        'deeplabv3plus_mobilenetv3': network.deeplabv3plus_mobilenetv3
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=16, pretrained_backbone=False)

    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("模型路径错误！")
        exit()



    transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    os.makedirs('test_results', exist_ok=True)

    import time
    time_list = []
    with torch.no_grad():
        model = model.eval()
        print(image_files)
        for img_path in tqdm(image_files):
            img_name = os.path.basename(img_path).split('.')[0]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0) # To tensor of NCHW
            img = img.to(device)

            t1 = time.time()
            pred = model(img)
            if opts.model=='deeplabv3plus_mobilenetv3':
                pred = pred['out']
            time_list.append(time.time() - t1)

            pred = pred.max(1)[1].cpu().numpy()[0] # HW
            colorized_preds = decode_fn(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)
          
            colorized_preds.save(os.path.join('test_results', img_name+'.png'))
    print('平均耗时:', sum(time_list)/len(time_list))
if __name__ == '__main__':
    main()
