import os
import torch
import torch.nn as nn
from internvl.train.dataset import build_transform, dynamic_preprocess
from internvl.model.internvl_chat import InternVisionModel, InternVLChatModel
from torchvision.utils import make_grid
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
QUAD_START_TOKEN = '<quad>'
QUAD_END_TOKEN = '</quad>'
REF_START_TOKEN = '<ref>'
REF_END_TOKEN = '</ref>'
BOX_START_TOKEN = '<box>'
BOX_END_TOKEN = '</box>'

def load_model(config, state_dict):
    vision_model = InternVisionModel(config.vision_config)
    vit = InternVLChatModel(config, vision_model).to(torch.bfloat16)
    vit.load_state_dict(state_dict, strict=False)
    tok_embeddings = nn.Embedding(config.llm_config.vocab_size, config.llm_config.hidden_size, 2).to(torch.bfloat16)
    tok_embeddings.weight = nn.Parameter(state_dict['language_model.model.tok_embeddings.weight'])
    return vit, tok_embeddings

def load_image(image_path):
    transform = get_transform(is_train=False, image_size=448)
    image = Image.open(image_path).convert('RGB')
    images, target_aspect_ratio = dynamic_preprocess(image, min_num=1, max_num=12,
                                image_size=448, use_thumbnail=True, return_ratio=True)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values).to(torch.bfloat16)
    return pixel_values, images, target_aspect_ratio

def get_similarity_map(sm, shape, min_max=True, threshold=0.2):
    B, N, H, W = sm.shape
    sm = sm.reshape(B, N, H*W)
    if min_max:
        # min-max norm
        sm = (sm - sm.min(2, keepdim=True)[0]) / (sm.max(2, keepdim=True)[0] - sm.min(2, keepdim=True)[0])
    else:
        sm = sm > threshold
        sm = sm.float()
    # reshape
    sm = sm.reshape(B, N, H, W).float()
    # interpolate
    sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear')    
    return sm


def get_transform(is_train, image_size):
    # Build transformation function
    transform = build_transform(is_train=is_train, input_size=image_size,
                                pad2square=False, normalize_type='imagenet')
    return transform

def post_process(vit_embeds, target_aspect_ratio):
    h = w = int(vit_embeds.shape[1] ** 0.5)
    vit_embeds_local = vit_embeds[:-1].reshape(-1, h, w, 4096).permute(0, 3, 1, 2)
    vit_embeds_local = make_grid(vit_embeds_local, nrow=target_aspect_ratio[0], padding=0, normalize=False)
    vit_embeds_local = vit_embeds_local.permute(1,2,0)
    H, W, C = vit_embeds_local.shape
    vit_embeds_local = vit_embeds_local.reshape(H*W, C)
    return vit_embeds_local, (H, W)

def generate_similiarity_map(images, attn_map, all_bpe_strings, out_dir, target_aspect_ratio=(1,1), image_size=448):
    if isinstance(images, list):
        images_vis = torch.stack([T.ToTensor()(image) for image in images[:-1]])
        images_vis = make_grid(images_vis, nrow=target_aspect_ratio[0], padding=0, normalize=False)
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
    else:
        images_vis = T.ToTensor()(images)
        target_height = images.size[1]
        target_width = images.size[0]
    
    attn_norm = get_similarity_map(attn_map.unsqueeze(0), (target_height, target_width), min_max=True, threshold=0.2)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Draw similarity map
    images_vis = (images_vis.permute(1,2,0).cpu().numpy() * 255).astype('uint8')
    for b in range(attn_norm.shape[0]):
        for n in range(attn_norm.shape[1]):
            vis = (attn_norm[b, n, :, :].float().detach().cpu().numpy() * 255).astype('uint8')
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            vis = images_vis * 0.4 + vis * 0.6
            vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
            plt.imshow(vis)
            plt.axis('off')
            plt.savefig(f'./{out_dir}/{all_bpe_strings[n]}.jpg', bbox_inches='tight', dpi=600)
            plt.close()
