from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetConfig:
    def __init__(self, in_channels, layers):
        self.in_channels = in_channels
        self.layers = layers


class Resnet18_2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.acv = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride > 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, stride=stride, kernel_size=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.acv(x)
        x = self.conv2(x)
        x = self.bn2(x)
        identity = self.downsample(identity)
        x = identity + x
        x = self.acv(x)
        return x


class block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv2d_1 = Resnet18_2D(
            in_channels=in_channels, out_channels=out_channels, stride=stride
        )
        self.conv2d_2 = Resnet18_2D(
            in_channels=out_channels, out_channels=out_channels, stride=1
        )

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)

        return x


# use resnet-encoder 18
class encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.downsample = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.layers[0],
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(config.layers[0])
        self.layer1 = block(
            in_channels=config.layers[0], out_channels=config.layers[0], stride=1
        )
        self.layer2 = block(
            in_channels=config.layers[0], out_channels=config.layers[1], stride=2
        )
        self.layer3 = block(
            in_channels=config.layers[1],
            out_channels=config.layers[2],
            stride=2,
        )
        self.layer4 = block(
            in_channels=config.layers[2],
            out_channels=config.layers[3],
            stride=2,
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class Upblock_2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv_up1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            output_padding=(stride - 1),
        )
        self.acv = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv_up2 = nn.ConvTranspose2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride > 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    stride=stride,
                    kernel_size=1,
                    bias=False,
                    output_padding=(stride - 1),
                ),
                nn.BatchNorm2d(out_channels),
            )

        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = x
        x = self.conv_up1(x)
        x = self.bn1(x)
        x = self.acv(x)
        x = self.conv_up2(x)
        x = self.bn2(x)
        identity = self.downsample(identity)
        x = identity + x
        x = self.acv(x)
        return x


class decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layers = list(reversed(config.layers))
        self.layer1 = Upblock_2D(
            in_channels=layers[0], out_channels=layers[1], stride=2
        )
        self.layer2 = Upblock_2D(
            in_channels=layers[1], out_channels=layers[2], stride=2
        )
        self.layer3 = Upblock_2D(
            in_channels=layers[2],
            out_channels=layers[3],
            stride=2,
        )
        self.layer4 = Upblock_2D(
            in_channels=layers[3],
            out_channels=layers[3],
            stride=2,
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=layers[3],
                out_channels=config.in_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                output_padding=1,
            ),
            nn.BatchNorm2d(config.in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.upsample(x)
        return x
