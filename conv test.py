import numpy as np
import torch
import torch.nn as nn


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            # nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True))
        # nn.ReLU(inplace=False))

    def forward(self, x):
        x = self.conv(x)
        return x


if __name__ =='__main__':
    # inputimg = torch.ones([1, 1, 6, 6])
    # conv = conv_block(1, 64)
    # out = conv(inputimg)    # 3 1 1 卷积不改变size
    # print(out.shape)
    # Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)    # pooling 2 2 使size减半
    # poolout = Maxpool1(out)
    # print(poolout.shape)

    mask = np.ones([6, 6])
    mask[1::2, ::2] = -1
    # mask[::2, 1::2] = -1
    print(mask)
