""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, pooling,in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            pooling,
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, pooling, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            pooling,
            nn.ReLU(inplace=True)
        )
        
        
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, pooling,bilinear=False):
        super(UNet, self).__init__()
        if pooling == 'max':
            pooling_layer = nn.MaxPool2d(2)
        elif pooling == 'avg':
            pooling_layer = nn.AvgPool2d(2)
        else:
            assert False, 'wrong pooling type for layoutnet!'

        self.n_channels = 3
        self.bilinear = bilinear

        self.inc = DoubleConv(3, 16)
        self.down1 = Down(pooling_layer,16, 32)
        self.down2 = Down(pooling_layer,32, 64)
        #self.down3 = Down(pooling_layer,54, 64)
        factor = 2 if bilinear else 1
        self.down3 = Down(pooling_layer,64, 128 // factor)
        self.up1 = Up(128, 64 // factor, bilinear)
        self.up2 = Up(64, 32 // factor, bilinear)
        self.up3 = Up(32, 16 // factor, bilinear)
        #self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(pooling_layer,16, 1)

    def forward(self, x):
        #print('x:',x.shape)
        x1 = self.inc(x)
        #print('x1:',x1.shape)
        x2 = self.down1(x1)
        #print('x2:',x2.shape)
        x3 = self.down2(x2)
        #print('x3:',x3.shape)
        x4 = self.down3(x3)
        #print('x4:',x4.shape)
        #x5 = self.down4(x4)
        #print('x5:',x5.shape)
        x = self.up1(x4, x3)
        #print('rx4:',x.shape)
        x = self.up2(x, x2)
        #print('rx3:',x.shape)
        x = self.up3(x, x1)
        #print('rx2:',x.shape)
        #x = self.up4(x, x1)
        #print('rx1:',x.shape)
        x= self.outc(x)
        #print('out',x.shape)
        return x