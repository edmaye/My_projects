# 使用说明

## 预测

- mobilenet

```bash
python .\predict.py --input test_images --model deeplabv3plus_mobilenet --ckpt checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth
```

- resnet101

```bash
python .\predict.py --input test_images --ckpt checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar --model deeplabv3plus_resnet101
```



参数说明

- --input后为要预测的图像路径（或者某个文件夹路径，把所有图片放在此文件夹中即可）

- ckpt为模型权重文件

- model为模型backbone类型

  ckpt和model两者要对应好，比如说用mobilenet的模型就得指定deeplabv3plus_mobilenet

## 重新训练

首先下载cityscapes数据集，放在data文件夹下。

训练指令（以resnet50作为backbone为例）：
```bash
python main.py --model deeplabv3plus_resnet50 --lr 0.1  --crop_size 768 --batch_size 8 --data_root ./datasets/data/cityscapes 
```

需要注意batch_size，不能超过gpu显存限制

如果需要在之前的模型基础上接着训练的话，加入以下两个参数

```bash
python main.py ... --ckpt YOUR_CKPT_PATH --continue_training
```


