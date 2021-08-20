import os
import json
import argparse
import matplotlib.pyplot as plt

from model import shufflenet_v2_x1_0
from models.mobilenetv1 import MobileNet
from models.mobilenetv2 import MobileNetV2
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import torch
import os
import sys
import json
import pickle
import random
from utils import read_split_data

import matplotlib.pyplot as plt


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = "./Demo/colonn7.jpeg"
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    img = img.convert("RGB")
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = MobileNetV2(num_classes=8).to(device)
    # load model weights
    model_weight_path = "./weights/model-epoch-29-acc-0.9599109131403119.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default="./Srinivasan")
    opt = parser.parse_args()
    main(opt)

