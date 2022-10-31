import glob
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from utils.MPResNet import MPResNet

class seg_qyl(nn.Module):
    def __init__(self, model_name, n_class):
        super().__init__()
        self.model = smp.UnetPlusPlus(# UnetPlusPlus
                encoder_name=model_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=n_class,                      # model output channels (number of classes in your dataset)
            )
    def forward(self, x):
        #with autocast():
        x = self.model(x)
        return x
if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    # 加载网络，图片单通道，分类为1。
    # net = Net(3, 1)
    model_name = 'efficientnet-b7'  # xception
    n_class = 1
    net = seg_qyl(model_name, n_class).cuda()

    # net = MPResNet(3, 1).cuda()
    # 将网络拷贝到deivce中
    # net.to(device=device)
    # 加载模型参数D:\SAR\Unet++\outputs\new\efficientnet-b7\ckptswa200\checkpoint-epoch44_fwiou98.49.pth
    net.load_state_dict(torch.load(r'D:\SAR\Unet++\outputs\crop_full\efficientnet-b7\checkpoint-epoch92_fwiou99.51.pth')['state_dict'])
    torch.save(net.state_dict(), 'cpts/crop-b7.pth', _use_new_zipfile_serialization=False)
    print('success!')