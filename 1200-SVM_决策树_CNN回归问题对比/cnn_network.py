from torch import nn
import torch


class conv_block(nn.Module):
    """
        通用卷积模块：卷积+BN层+ReLU层+池化层
    """
    def __init__(self,input,output):
        super(conv_block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input, output, kernel_size=3),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    def forward(self,x):
        return self.layer(x)

class CNN(nn.Module):
    """
        网络结构：输出 Nx2的矩阵
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = conv_block(3,32)
        self.layer2 = conv_block(32,48)
        self.layer3 = conv_block(48,64)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__=='__main__':
    model = CNN()
    x = torch.zeros((16,3,512,512))
    out = model(x)
    print(out.shape)