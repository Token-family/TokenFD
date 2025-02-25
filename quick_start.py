import os
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from internvl.model.internvl_chat import InternVLChatConfig, InternVisionModel
from utils import  post_process, generate_similiarity_map, load_model, load_image

checkpoint = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/guantongkun/work_dirs/8B_all_32/checkpoint-370000'
image_path = './demo_images/0000000.png'
input_query = '11/12/2020'
out_dir = 'results'

if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

"""loading model, tokenizer, tok_embeddings """
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, use_fast=False)
config = InternVLChatConfig.from_pretrained(checkpoint)
state_dict = load_file(checkpoint+'/model.safetensors')
model, tok_embeddings = load_model(config, state_dict)
# model = model.cuda()
# tok_embeddings = tok_embeddings.cuda()

"""loading image """
pixel_values, images, target_aspect_ratio = load_image(image_path)
 

"""loading query texts """
if input_query[0] in '!"#$%&\'()*+,-./0123456789:;<=>?@^_{|}~0123456789':
    input_ids = tokenizer(input_query)['input_ids'][1:]
else:
    input_ids = tokenizer(' '+input_query)['input_ids'][1:]
input_ids = torch.Tensor(input_ids).long().to(model.device)
input_embeds = tok_embeddings(input_ids).clone()
all_bpe_strings = [tokenizer.decode(input_id) for input_id in input_ids]


"""Obtaining similarity """
vit_embeds = model.forward_tokenocr(pixel_values.to(model.device)) #(vit_batch_size, 16*16, 2048)
vit_embeds_local, resized_size = post_process(vit_embeds, target_aspect_ratio)
token_features = vit_embeds_local / vit_embeds_local.norm(dim=-1, keepdim=True)
input_embedings = input_embeds / input_embeds.norm(dim=-1, keepdim=True)
similarity = input_embedings @ token_features.t()
attn_map = similarity.reshape(len(input_embedings), resized_size[0], resized_size[1])

"""generate map locally """
generate_similiarity_map(images, attn_map, all_bpe_strings, out_dir, target_aspect_ratio)


"""user command """
# python quick_start.py
