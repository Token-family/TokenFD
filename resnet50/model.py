import torch
import torch.nn.functional as F
from torch import nn
from .backbone import build_backbone
import pdb
import numpy as np
from typing import Optional

class TokenOCR(nn.Module):
    def __init__(self, backbone):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes

        """
        super().__init__()
        self.language_embedding = nn.Embedding(92553, 2048, padding_idx=2)
        for p in self.parameters():
            p.requires_grad = False

        self.backbone = backbone
        init_tau=np.log(10)
        init_b=-2.71
        self.t_prime = nn.Parameter(torch.ones([]) * init_tau)
        self.b = nn.Parameter(torch.ones([]) * init_b)
        self.kb = True
        self.upsample = nn.Sequential(
                    nn.ConvTranspose2d(
            in_channels=2048,
            out_channels=512,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        ),
        nn.SyncBatchNorm(512),
        nn.ConvTranspose2d(
            in_channels=512,
            out_channels=512,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        ),
        nn.SyncBatchNorm(512),
        )
        self.ocr_mlp = nn.Sequential(
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048)
        )

    def forward_tokenocr(self, pixel_values):
        vit_embeds = self.backbone(pixel_values)
        vit_embeds = vit_embeds['0']
        h, w = vit_embeds.shape[2], vit_embeds.shape[3]
        vit_embeds = self.upsample(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-2] * vit_embeds.shape[-1])
        vit_embeds = self.ocr_mlp(vit_embeds.permute(0, 2, 1))
        return vit_embeds, (h*4, w*4)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    backbone = build_backbone(args)
    model = TokenOCR(backbone)
    return model
