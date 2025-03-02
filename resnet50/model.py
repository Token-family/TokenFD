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

    def forward(self, 
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            image_flags: Optional[torch.LongTensor] = None,
            mask_values: Optional[torch.LongTensor] = None,
            masks_flags: Optional[torch.LongTensor] = None,
            mask_nums: Optional[torch.LongTensor] = None,
            ):
        image_flags = image_flags.squeeze(-1)
        try:
            input_embeds = self.language_embedding(input_ids).clone()
        except:
            print('error'*1000)
            import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        vit_embeds, vit_embeds_shape = self.extract_feature_custom(pixel_values) #(vit_batch_size, 16*16, 2048)
        nb, nl, nd = vit_embeds.shape
        h, w = vit_embeds_shape
        vit_embeds = vit_embeds.reshape(nb, h, w, nd)
        vit_embeds = vit_embeds.split(list(image_flags)) #[(vit_batch_size / B, h, w, C)]*B
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        try:
            assert sum(image_flags) == mask_values.shape[0]
        except:
            print((mask_values.shape, image_flags, mask_nums))
        
        mask_values = torch.nn.functional.interpolate(mask_values.float(), size=(h, w), mode='bilinear', align_corners=False) #(128, 128)
        masks = mask_values.split(list(image_flags)) #[(vit_batch_size / B, N, 448, 448)]*B

        
        masks_flags = masks_flags.chunk(B)
        token_features = []
        input_embedings = []
        masked_input_ids = []
        masked_zero_bools = []
        for i, vit_embed in enumerate(vit_embeds):
            current_token = masks_flags[i].sum()
            mask = masks[i]
            limit_num = mask.shape[1]
            mask = mask.permute(1,0,2,3).reshape(limit_num, -1) > 0
            max_cluster_index = mask.sum(-1)
            zero_bool = max_cluster_index != 0
            # import pdb; pdb.set_trace()
            mask[~zero_bool] = 1 #for addressing bflost16 bug
            new_max_cluster_index = mask.sum(-1)
            mask = mask / new_max_cluster_index.unsqueeze(-1)
            token_feature = torch.matmul(mask.to(vit_embed), vit_embed.reshape(-1, vit_embed.shape[-1]))
            token_features.extend(token_feature)
            input_embedings.extend(input_embeds[i, :])
            masked_input_ids.extend(input_ids[i, zero_bool])
            masked_zero_bools.append(zero_bool)

        masked_zero_bools = torch.cat(masked_zero_bools)
        token_features = torch.stack(token_features)
        input_embedings= torch.stack(input_embedings)

        loss2 = F.mse_loss(token_features, input_embedings, reduction='none')[masked_zero_bools].sum(1).sqrt().mean()
        token_features = token_features / token_features.norm(dim=1, keepdim=True)
        input_embedings = input_embedings / input_embedings.norm(dim=1, keepdim=True)
        # cosine similarity as logits
        similarity = F.cosine_similarity(token_features, input_embedings, dim=1)
        loss1 = (1 - similarity[masked_zero_bools]).mean()
        # loss_d = loss1 + loss2
        # if rank == 0:
            # print(f'loss1:{loss_d}')

        ###siglip
        # masked_input_ids = torch.stack(masked_input_ids)
        # label_matrix = (masked_input_ids.unsqueeze(0) == masked_input_ids.unsqueeze(1)).int()
        # label_matrix = 2 * label_matrix - 1
        # if self.kb:
        #     logits = (input_embedings[masked_zero_bools] @ token_features[masked_zero_bools].t()) * self.t_prime.to(input_embedings.device).exp() + self.b.to(input_embedings.device)
        # else:
        #     logits = (input_embedings[masked_zero_bools] @ token_features[masked_zero_bools].t()) * self.t_prime.to(input_embedings.device).exp() - 8.9375
        # loss_s = -torch.sum(F.logsigmoid(label_matrix * logits)) / logits.shape[0]
        # if rank == 0:
        #     print(f'loss2:{loss_s}')
        return loss1, loss2

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
