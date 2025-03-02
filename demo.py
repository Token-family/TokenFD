import os
import argparse
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
from transformers import AutoTokenizer
from resnet50 import build_model
from utils import generate_similiarity_map, get_transform, post_process, load_tokenizer, build_transform_R50
from internvl.train.dataset import dynamic_preprocess, build_transform
from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel, InternVisionModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='VIT')
    parser.add_argument('--checkpoint', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mt-ocr/guantongkun/VFM_try/processed_models/TokenOCR_4096_English_seg')
    parser.add_argument('--image_path', type=str, default='demo_images/00000002.jpg')
    parser.add_argument('--text_input', type=str, default='they however are not')
    parser.add_argument('--out_dir', type=str, default='visualization_dir')
    parser.add_argument('--device', default='cuda:0', help='device to use for training / testing')
    args = parser.parse_args()

    device =torch.device(args.device)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    if args.type == 'R50':
        """loading model, tokenizer, tok_embeddings """
        tokenizer_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mt-ocr/guantongkun/VFM_try/processed_models/r50/tokenizer_path'
        tokenizer = load_tokenizer(tokenizer_path)
        model = build_model(args).eval()
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model'])
        model.to(device)
        """loading image """
        transform = build_transform_R50(normalize_type='imagenet')
        images = Image.open(args.image_path).convert('RGB')
        pixel_values = torch.stack([transform(images)]).to(device) 
        target_aspect_ratio = (1,1)

    elif args.type == 'VIT':
        """loading model, tokenizer, tok_embeddings """
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
        model = InternVLChatModel.from_pretrained(args.checkpoint, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,load_in_8bit=False, load_in_4bit=False).eval()
        model = model.to(device)
        """loading image """
        transform = get_transform(is_train=False, image_size=model.config.force_image_size)
        image = Image.open(args.image_path).convert('RGB')
        images, target_aspect_ratio = dynamic_preprocess(image, min_num=1, max_num=12,
                                    image_size=model.config.force_image_size, use_thumbnail=model.config.use_thumbnail, return_ratio=True)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values).to(torch.bfloat16).to(model.device)  

    else:
        raise f'not support type {args.type}'

    """Load query texts """
    text_input=args.text_input
    if text_input[0] in '!"#$%&\'()*+,-./0123456789:;<=>?@^_{|}~0123456789':
        input_ids = tokenizer(text_input)['input_ids'][1:]
    else:
        input_ids = tokenizer(' '+text_input)['input_ids'][1:]
    input_ids = torch.Tensor(input_ids).long().to(device)
    if args.type == 'R50':
        input_embeds = model.language_embedding(input_ids).clone()
    elif args.type == 'VIT':
        input_embeds = model.tok_embeddings(input_ids).clone()

    """Obtaining similarity """
    vit_embeds, size1 = model.forward_tokenocr(pixel_values.to(device)) #(vit_batch_size, 16*16, 2048)
    vit_embeds_local, size2 = post_process(vit_embeds, target_aspect_ratio, args.type)
    token_features = vit_embeds_local / vit_embeds_local.norm(dim=-1, keepdim=True)
    input_embedings = input_embeds / input_embeds.norm(dim=-1, keepdim=True)
    similarity = input_embedings @ token_features.t()
    resized_size = size1 if size1 is not None else size2
    attn_map = similarity.reshape(len(input_embedings), resized_size[0], resized_size[1])

    """generate map locally """
    all_bpe_strings = [tokenizer.decode(input_id) for input_id in input_ids]
    generate_similiarity_map(images, attn_map, all_bpe_strings, args.out_dir, target_aspect_ratio)
 

"""use guide
for r50
python demo.py --type R50 --checkpoint /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mt-ocr/guantongkun/VFM_try/processed_models/r50/checkpoint.pth
for VIT
python demo.py --type VIT --checkpoint /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mt-ocr/guantongkun/VFM_try/processed_models/TokenOCR_2048_Binlinual_seg
python demo.py --type VIT --checkpoint /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mt-ocr/guantongkun/VFM_try/processed_models/TokenOCR_4096_English_seg
"""