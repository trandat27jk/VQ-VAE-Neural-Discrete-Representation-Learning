from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PixelCnn_config:
    codeblock_dim: int = None
    layers: int = None
    kernel_size: int = None
    in_channels: int = None
    num_codeblocks: int = None
    cobdeblock_h: int = None
    cobdeblock_k: int = None


class MaskedConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels=int,
        out_channels=int,
        kernel_size=int,
        stride=int,
        padding=int,
        mask_type=str,
        gated=bool,
    ):
        actual_out_channels = 2 * out_channels if gated else out_channels
        super().__init__(
            in_channels,
            actual_out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        center = int(kernel_size // 2)
        base_mask = torch.ones(kernel_size, kernel_size)
        vertical_mask = base_mask.clone()
        vertical_mask[center + 1 :, :] = 0
        horizontal_mask = base_mask.clone()
        if mask_type == "horizontal-A":
            horizontal_mask[:, center:] = 0
        elif mask_type == "horizontal-B":
            horizontal_mask[:, center + 1 :] = 0
        mask = vertical_mask * horizontal_mask
        self.gated = gated
        if gated:
            mask = (
                mask.unsqueeze(0)
                .unsqueeze(0)
                .expand(2 * out_channels, in_channels, kernel_size, kernel_size)
            )
        else:
            mask = (
                mask.unsqueeze(0)
                .unsqueeze(0)
                .expand(out_channels, in_channels, kernel_size, kernel_size)
            )
        self.register_buffer("mask", mask)

    def forward(self, x):
        masked_weight = self.weight * self.mask
        output = F.conv2d(
            x, masked_weight, bias=self.bias, stride=self.stride, padding=self.padding
        )
        if self.gated:
            output1, output2 = torch.chunk(output, 2, dim=1)
            return torch.tanh(output1) * torch.sigmoid(output2)
        return output


class PixelCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleDict(
            dict(
                first_layer=MaskedConv2d(
                    in_channels=config.in_channels,
                    out_channels=config.codeblock_dim,
                    kernel_size=config.kernel_size,
                    stride=1,
                    padding=2,
                    gated=True,
                ),
                blocks=nn.ModuleList(
                    [
                        MaskedConv2d(
                            in_channels=config.in_channels,
                            out_channels=config.codeblock_dim,
                            kernel_size=config.kernel_size,
                            stride=1,
                            padding=2,
                            gated=True,
                        )
                        for i in range(config.layers)
                    ]
                ),
                final_layer=MaskedConv2d(
                    in_channels=config.in_channels,
                    out_channels=config.in_channels,
                    kernel_size=5,
                    stride=config.kernel_size,
                    padding=2,
                    gated=False,
                ),
            )
        )
        self.config = config
        self.to_out = nn.Linear(config.codeblock_dim, config.num_codeblocks)

    def forward(self, x):
        x = self.layers["first_layer"](x)
        for block in self.layers["blocks"]:
            x = block(x)
        x = self.layers["final_layer"](x)
        x = self.to_out(x)
        return x

    def generation(self, codeblock):
        images = torch.zeros_like((self.config.codeblock_h, self.config.codeblock_w))
        for i in range(self.config.codeblock_h):
            for j in range(self.config.codeblock_k):
                out = self[codeblock]
