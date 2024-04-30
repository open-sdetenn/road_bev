"""
open-sdetenn bev model v1 (used in V2!) Intended to be cheap and simple for showcasing the potential use case :) Torch implementation of the example model from carla, with simplified architecture.
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, needed_labels):
        super(Model, self).__init__()

        kernel_size = [8, 6, 4]
        strides = 2
        input_channels = 96
        leakyrelu_alpha = 0.2
        output_channels = len(needed_labels)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=8, stride=strides),
            nn.LeakyReLU(leakyrelu_alpha)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=6, stride=strides),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(leakyrelu_alpha)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=strides),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leakyrelu_alpha)
        )

        self.decoder6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=6),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(leakyrelu_alpha)
        )
        self.decoder7 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=6),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(leakyrelu_alpha)
        )
        self.decoder8 = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=8),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.float()
        encoder1 = self.encoder1(x)
        encoder2 = self.encoder2(encoder1)
        encoder3 = self.encoder3(encoder2)

        decoder6 = self.decoder6(encoder3)
        decoder7 = self.decoder7(decoder6)
        decoder8 = self.decoder8(decoder7)

        return decoder8
