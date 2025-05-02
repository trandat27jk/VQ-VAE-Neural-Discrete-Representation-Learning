from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ResnetConfig, decoder, encoder


@dataclass
class Vq_config:
    num_codeblocks: int = 1024


class VQ_VAE(nn.Module):
    def __init__(self, config, image_config):
        super().__init__()
        self.config = config
        self.encoder = encoder(image_config)
        self.decoder = decoder(image_config)
        self.codeblocks = nn.Parameter(
            torch.randn(config.num_codeblocks, image_config.layers[-1])
        )

    def forward(self, x):
        encoder_out = self.encoder(x)
        B, C, H, W = encoder_out.shape
        reshape_out = encoder_out.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
        diff = torch.sum(
            (reshape_out.unsqueeze(1) - self.codeblocks.unsqueeze(0)) ** 2, dim=-1
        )
        nearest_codeblocks = torch.argmin(diff, dim=1)
        quantized = self.codeblocks[nearest_codeblocks]
        quantized = reshape_out + (quantized - reshape_out).detach()
        quantized_reshape = quantized.view(B, H, W, C).permute(0, 3, 1, 2)
        decoder_out = self.decoder(quantized_reshape)
        return {"output": decoder_out, "ze": reshape_out, "e": quantized}


def vq_vae_loss(output, target, ze, e, beta):
    recon_loss = F.mse_loss(output, target)
    l2_loss_1 = F.mse_loss(ze.detach(), e)
    l2_loss_2 = beta * F.mse_loss(ze, e.detach())
    return recon_loss + l2_loss_1 + l2_loss_2
