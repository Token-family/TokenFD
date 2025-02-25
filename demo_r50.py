import os
import argparse
import cv2
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
from transformers import AutoTokenizer
from resnet50 import build_model
from utils import generate_similiarity_map, IMAGENET_MEAN, IMAGENET_STD, IMG_CONTEXT_TOKEN, IMG_START_TOKEN, IMG_END_TOKEN, QUAD_START_TOKEN, \
    QUAD_END_TOKEN, REF_START_TOKEN, REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN


def build_transform(normalize_type='imagenet'):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/guantongkun/PreTraining/R50/checkpoint.pth')
    parser.add_argument('--path', type=str, default='demo_images/00000002.jpg')
    parser.add_argument('--text_input', type=str, default='they however are not')
    parser.add_argument('--out_dir', type=str, default='visualization_dir')
    parser.add_argument('--device', default='cpu', help='device to use for training / testing')
    parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
    args = parser.parse_args()

    device =torch.device(args.device)

    # Load tokenizer model
    tokenizer_path = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/guantongkun/PreTraining/SPTSv2/tokenizer_path'
    tokenizer = load_tokenizer(tokenizer_path)

    # Load pretrained model
    model = build_model(args)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model'])
    model.to(device)

    # Load image
    transform = build_transform(normalize_type='imagenet')
    image = Image.open(args.path).convert('RGB')
    pixel_values = torch.stack([transform(image)]).to(device) 

    # Load query texts
    text_input=args.text_input
    if text_input[0] in '!"#$%&\'()*+,-./0123456789:;<=>?@^_{|}~0123456789':
        input_ids = tokenizer(text_input)['input_ids'][1:]
    else:
        input_ids = tokenizer(' '+text_input)['input_ids'][1:]
    input_ids = torch.Tensor(input_ids).long().to(device)

    # get similarity map             
    vit_embeds, paded_size = model.extract_feature_custom(pixel_values)
    vit_embeds = vit_embeds.reshape(-1, vit_embeds.shape[-1])
    input_embeds = model.language_embedding(input_ids).clone()
    token_features = vit_embeds / vit_embeds.norm(dim=-1, keepdim=True)
    input_embedings = input_embeds / input_embeds.norm(dim=-1, keepdim=True)
    similarity = input_embedings @ token_features.t() #* torch.tensor([3.75]).to(input_embedings.device).exp() - 8.9375
    attn_map = similarity.reshape(len(input_embedings), paded_size[0], paded_size[1])

    # generate map locally
    all_bpe_strings = [tokenizer.decode(input_id) for input_id in input_ids]
    generate_similiarity_map(image, attn_map, all_bpe_strings, args.out_dir)

