from .modeling_intern_vit import InternVisionEncoderLayer
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenvlVisionExtra(nn.Module):
    def __init__(self, config):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, 0.1, 24)]
        self.vit_layer = InternVisionEncoderLayer(config, dpr[-1]) 

        self.upsample = nn.Sequential(
                        nn.ConvTranspose2d(
                        in_channels=1024,
                        out_channels=512,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False),
                        # nn.BatchNorm2d(512),
                        nn.SyncBatchNorm(512),
                        # 第二层反卷积：进一步上采样到目标分辨率
                        nn.ConvTranspose2d(
                            in_channels=512,
                            out_channels=1024,
                            kernel_size=4,
                            stride=2,
                            padding=1,
                            bias=False),
                        # nn.BatchNorm2d(1024),
                        nn.SyncBatchNorm(1024),)    

    def forward(self, x):
        vit_embeds = self.vit_layer(x)
        vit_embeds = vit_embeds[:, 1:, :]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.permute(0,2,1).reshape(vit_embeds.shape[0], -1, h, w)
        vit_embeds = self.upsample(vit_embeds) # BCHW

        return vit_embeds


class TokenReducer(nn.Module):
    def __init__(self):
        super().__init__()
        self.foreground_tokens = nn.Embedding(1, 1024)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, 4, 4).unfold(3, 4, 4).contiguous().view(B, C, H // 4, W // 4, -1).view(B, C, -1, 16)
        x = x.permute(0,2,3,1)
        hyper_in = self.foreground_tokens.weight.t().unsqueeze(0).unsqueeze(0)
        pred_result = (x @ hyper_in).view(B, -1, 16)
        x = F.softmax(pred_result, dim=-1).unsqueeze(-1) * x
        x = x.sum(dim=2)
        x = x.reshape(B, H // 4, W // 4, C)

        return x



def change_state_dict(st):

    new_st = {}
    for k, v in st.items():
        new_k = k.replace('backbone_encoder.layers.0.','vit_layer.').replace('ocrmlp_', 'ocr_mlp.').replace('upsample_', 'upsample.').replace('v_encoder.layers.0.','vit_layer.').replace('u_', 'upsample.')
        if new_k.startswith('m_'):
            new_k = 'ocr_mlp.' + new_k[2:]
        new_st[new_k] = v

    return new_st