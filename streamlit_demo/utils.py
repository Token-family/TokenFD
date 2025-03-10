import os
import torch
import torch.nn as nn
from internvl.train.dataset import build_transform, dynamic_preprocess
from internvl.model.internvl_chat import InternVisionModel, InternVLChatModel
from torchvision.utils import make_grid
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
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

def build_transform_R50(normalize_type='imagenet'):
    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == 'clip':
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == 'siglip':
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise NotImplementedError
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def load_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=False)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = 8192
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    return tokenizer

def get_transform(is_train, image_size):
    # Build transformation function
    transform = build_transform(is_train=is_train, input_size=image_size,
                                pad2square=False, normalize_type='imagenet')
    return transform

def post_process(vit_embeds, target_aspect_ratio, model_type='VIT'):
    if model_type in ["TokenFD-4096-English-seg", "TokenFD-2048-Bilingual-seg"]:
        h = w = int(vit_embeds.shape[1] ** 0.5)
        c = vit_embeds.shape[-1]
        # vit_embeds_local = vit_embeds[:-1].reshape(-1, h, w, c).permute(0, 3, 1, 2)
        if vit_embeds.shape[0] == 1:
            vit_embeds_local = vit_embeds.reshape(-1, h, w, c).permute(0, 3, 1, 2)
        else:
            vit_embeds_local = vit_embeds[:-1].reshape(-1, h, w, c).permute(0, 3, 1, 2)
        vit_embeds_local = make_grid(vit_embeds_local, nrow=target_aspect_ratio[0], padding=0, normalize=False)
        vit_embeds_local = vit_embeds_local.permute(1,2,0)
        H, W, C = vit_embeds_local.shape
        vit_embeds_local = vit_embeds_local.reshape(H*W, C)
        return vit_embeds_local, (H, W)
    if model_type== 'R50':
        vit_embeds = vit_embeds.reshape(-1, vit_embeds.shape[-1])
        return vit_embeds, None

def generate_similiarity_map(images, attn_map, all_bpe_strings, vis_list, target_aspect_ratio=(1,1), src_image_size=(1024, 1024), image_size=448):
    if isinstance(images, list):
        # images_vis = torch.stack([T.ToTensor()(image) for image in images[:-1]])
        if len(images) == 1:
            images_vis = torch.stack([T.ToTensor()(image) for image in images])
        else:
            images_vis = torch.stack([T.ToTensor()(image) for image in images[:-1]])
        images_vis = make_grid(images_vis, nrow=target_aspect_ratio[0], padding=0, normalize=False)
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
    else:
        images_vis = T.ToTensor()(images)
        target_height = images.size[1]
        target_width = images.size[0]
    
    attn_norm = get_similarity_map(attn_map.unsqueeze(0), (target_height, target_width), min_max=True, threshold=0.2)


    # Draw similarity map
    print(images_vis.shape)
    images_vis = (images_vis.permute(1,2,0).cpu().numpy() * 255).astype('uint8')
    for b in range(attn_norm.shape[0]):
        for n in range(attn_norm.shape[1]):
            vis = (attn_norm[b, n, :, :].float().detach().cpu().numpy() * 255).astype('uint8')
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            vis = images_vis * 0.4 + vis * 0.6
            vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
            vis = cv2.resize(vis, src_image_size)
            vis_list.append(vis)  # Add each visualization to the list

    return vis_list


def load_model_and_tokenizer_customed(checkpoint):
    kwargs = {}
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False)
    model = InternVLChatModel.from_pretrained(
        checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
        load_in_8bit=False, load_in_4bit=False, **kwargs).eval()
    del model.language_model.model.layers
    del model.language_model.output
    model = model.cuda()
    return model, tokenizer