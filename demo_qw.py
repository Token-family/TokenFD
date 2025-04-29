###tokenfd_torch25
import sys
import os
import torch
import argparse
from PIL import Image
from utils import generate_similiarity_map_qw
from transformers import AutoProcessor, AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
from transformers import Qwen2_5_VLProcessor
sys.path.append('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mt-ocr/guantongkun/ms_swift')
from swift.llm.model.model.marten import Marten_Q

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id_or_path', type=str, 
                        default='./guantongkun/work_dirs/tokenfd_qw/7B/checkpoint-60000')
    parser.add_argument('--image_path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mt-ocr/guantongkun/VFM/demo_images/0000000.png')
    parser.add_argument('--str', type=str, default='11/12/2020')
    parser.add_argument('--out_dir', type=str, default='results')
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id_or_path, trust_remote_code=True)
    model = Marten_Q.from_pretrained(args.model_id_or_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
    model = model.cuda()
    processor = AutoProcessor.from_pretrained(args.model_id_or_path, trust_remote_code=True)

    input_image = Image.open(args.image_path).convert('RGB')
    resized_height, resized_width = smart_resize(
        input_image.height,
        input_image.width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=processor.image_processor.max_pixels,
    )
    inputs = processor(
        text=[args.str],
        images=input_image,
        # videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to('cuda')
    pixel_values = inputs['pixel_values'].to(torch.bfloat16)
    grid_thw = inputs['image_grid_thw']
    input_ids = inputs['input_ids']
    """loading query texts """
    input_embeds = model.model.embed_tokens(input_ids).clone()
    all_bpe_strings = [tokenizer.decode(input_id) for input_id in input_ids[0]]
    """loading visual image"""
    vit_embeds = model.customed_visual_forward(pixel_values, grid_thw)
    vit_embeds = model.ocr_mlp(vit_embeds)
    llm_size = model.model.embed_tokens.weight.shape[1]
    vit_embeds = vit_embeds.reshape(-1, 2, 2, llm_size)
    vit_embeds = vit_embeds.reshape(grid_thw[0,1]//2, grid_thw[0,2]//2, 2, 2, llm_size).permute(0, 2, 1, 3, 4).reshape(1, grid_thw[0,1], grid_thw[0,2], llm_size)


    """Obtaining similarity """
    vit_embeds = vit_embeds.reshape(-1, llm_size)
    input_embeds = input_embeds.reshape(-1, llm_size)
    # vit_embeds_local, resized_size = post_process(vit_embeds, target_aspect_ratio)
    token_features = vit_embeds / vit_embeds.norm(dim=-1, keepdim=True)
    input_embedings = input_embeds / input_embeds.norm(dim=-1, keepdim=True)
    similarity = input_embedings @ token_features.reshape(-1, llm_size).t()
    attn_map = similarity.reshape(len(input_embedings), resized_height//14, resized_width//14)

    """generate map locally """
    # import pdb; pdb.set_trace()
    resized_image = input_image.resize((resized_width, resized_height))
    generate_similiarity_map_qw(resized_image, attn_map, all_bpe_strings, args.out_dir)