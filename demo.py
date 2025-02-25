import torch
import torch.nn as nn
import os
import argparse
from PIL import Image
from internvl.model import load_model_and_tokenizer_customed
from internvl.train.dataset import dynamic_preprocess, build_transform
from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel, InternVisionModel
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from utils import get_transform, post_process, generate_similiarity_map, load_model
from safetensors.torch import load_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/guantongkun/work_dirs/8B_all_32/checkpoint-370000')
    parser.add_argument('--image_path', type=str, default='')
    parser.add_argument('--str', type=str, default='')
    parser.add_argument('--out_dir', type=str, default='results')
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    """loading model, tokenizer, tok_embeddings """
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, use_fast=False)
    config = InternVLChatConfig.from_pretrained(args.checkpoint)
    state_dict = load_file(args.checkpoint+'/model.safetensors')
    model, tok_embeddings = load_model(config, state_dict)
    model = model.cuda()
    tok_embeddings = tok_embeddings.cuda()

    """loading image """
    transform = get_transform(is_train=False, image_size=model.config.force_image_size)
    image = Image.open(args.image_path).convert('RGB')
    images, target_aspect_ratio = dynamic_preprocess(image, min_num=1, max_num=12,
                                image_size=model.config.force_image_size, use_thumbnail=model.config.use_thumbnail, return_ratio=True)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values).to(torch.bfloat16).to(model.device)   

    """loading query texts """
    input_ids = tokenizer(args.str)['input_ids'][1:]
    input_ids = torch.Tensor(input_ids).long().to(model.device)
    input_embeds = tok_embeddings(input_ids).clone()
    all_bpe_strings = [tokenizer.decode(input_id) for input_id in input_ids]


    """Obtaining similarity """
    vit_embeds = model.forward_tokenocr(pixel_values) #(vit_batch_size, 16*16, 2048)
    vit_embeds_local = post_process(vit_embeds, target_aspect_ratio)
    H, W, C = vit_embeds_local.shape
    vit_embeds_local = vit_embeds_local.reshape(H*W, C)
    token_features = vit_embeds_local / vit_embeds_local.norm(dim=-1, keepdim=True)
    input_embedings = input_embeds / input_embeds.norm(dim=-1, keepdim=True)
    similarity = input_embedings @ token_features.t()
    attn_map = similarity.reshape(len(input_embedings), H, W)

    """generate map locally """
    generate_similiarity_map(args, images, attn_map, all_bpe_strings, target_aspect_ratio)
    

    """user command """
    # python demo.py --path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/guantongkun/PreTraining/InternVL/downstream_tasks/0000000.png --str 11/12/2020
