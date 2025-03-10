import os
import argparse
import io
from typing import List

import numpy as np
import pypdfium2
import streamlit as st
from PIL import Image
import fitz
from base64 import b64encode
import os

from io import BytesIO
import torch
import os
import argparse
import sys
from PIL import Image
import numpy as np
import os
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from transformers import AutoTokenizer
from resnet50 import build_model
import cv2
from utils import generate_similiarity_map, get_transform, post_process, load_tokenizer, build_transform_R50
from utils import IMAGENET_MEAN, IMAGENET_STD, IMG_CONTEXT_TOKEN, IMG_START_TOKEN, IMG_END_TOKEN, QUAD_START_TOKEN, \
    QUAD_END_TOKEN, REF_START_TOKEN, REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN


from internvl.train.dataset import dynamic_preprocess, build_transform
from internvl.model.internvl_chat import InternVLChatConfig, InternVLChatModel, InternVisionModel

import base64
from io import BytesIO

# def pil_to_base64(pil_image):
#     buffered = BytesIO()
#     pil_image.save(buffered, format="PNG")
#     return base64.b64encode(buffered.getvalue()).decode()


# 将 PIL 图像转换为 Base64
def pil_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# 放大图像尺寸 3 倍
def resize_image(pil_image, scale_factor):
    width, height = pil_image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)



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

def pixel_unshuffle(x, scale_factor=4):
    h = w = int(x.shape[1] ** 0.5)
    n, l, c = x.size()
    x = x.reshape(n, h, w, c)
    x = x.repeat_interleave(scale_factor, dim=1).repeat_interleave(scale_factor, dim=2)
    return x

def extract_feature_custom(model, pixel_values, use_mlp=True):
    vit_embeds = model.vision_model(
        pixel_values=pixel_values,
        output_hidden_states=False,
        return_dict=True).last_hidden_state

    vit_embeds = vit_embeds[:, 1:, :] # (52, 1025, 1024)

    h = w = int(vit_embeds.shape[1] ** 0.5)
    vit_embeds = model.ocr_mlp(vit_embeds)
    vit_embeds = pixel_unshuffle(vit_embeds)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
    return vit_embeds




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
    sm = sm.reshape(B, N, H, W)
    # interpolate
    sm = sm.float()

    sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear')    
    return sm



parser = argparse.ArgumentParser(description="Run OCR on an image or PDF.")


try:
    args = parser.parse_args()
except SystemExit as e:
    print(f"Error parsing arguments: {e}")
    os._exit(e.code)

# def open_pdf(pdf_file):
#     stream = io.BytesIO(pdf_file.getvalue())
#     return pypdfium2.PdfDocument(stream)

@st.cache_data()
def get_page_image(pdf_file, page_num, dpi=250):
    doc = open_pdf(pdf_file)
    renderer = doc.render(
        pypdfium2.PdfBitmap.to_pil,
        page_indices=[page_num - 1],
        scale=dpi / 72,
    )
    png = list(renderer)[0]
    png_image = png.convert("RGB")
    return png_image

@st.cache_data()
def get_page_image2(pdf_file, page_num):
    pdfDoc = fitz.open(stream=io.BytesIO(pdf_file.getvalue()), filetype="pdf")
    zoom_x, zoom_y = None, None
    page = pdfDoc[page_num - 1]
    rotate = int(0)
    if zoom_x is None and zoom_y is None:
        x, y, w, h = page.bound()
        zoom_x = 2000 / w
        zoom_y = 2000 / w
    # 提取图片
    mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
    pix = page.get_pixmap(matrix=mat, colorspace='rgb', alpha=False)
    image = cv2.imdecode(np.frombuffer(pix.tobytes(), np.uint8), cv2.IMREAD_COLOR)

    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))



# 调试模式下使用固定的 PDF 文件
def load_fixed_pdf(file_path):
    with open(file_path, 'rb') as f:
        return io.BytesIO(f.read())

st.set_page_config(layout="wide")
col1, col2 = st.columns([.5, .5])



# 初始化 md_flag 如果不存在
if 'md_flag' not in st.session_state:
    st.session_state.md_flag = 0


if st.session_state.md_flag == 0:
    st.markdown("""
    <style>
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 50vh;
    }
    .content {
        max-width: 1100px;
        text-align: left;
    }
    </style>
    <div class="centered">
    <div class="content">
    <h1>欢迎使用Token级基座模型的Bpe Visualization Demo!</h1>
    <p> </p>
        <p> </p>
            <p> </p>
    <p>在这里您可以使用并查看模型实时生成bpe可视化结果。</p>
    </div>
    </div>
    """, unsafe_allow_html=True)


flag = 0
pageText_flag = 0



in_file = None
# 初始化 session_state 变量
if 'in_file0' not in st.session_state:
    st.session_state.in_file0 = None
if 'in_file1' not in st.session_state:
    st.session_state.in_file1 = None
if 'in_file2' not in st.session_state:
    st.session_state.in_file2 = None

if 'sam0' not in st.session_state:
    st.session_state.sam0 = 0
if 'sam1' not in st.session_state:
    st.session_state.sam1 = 0
if 'sam2' not in st.session_state:
    st.session_state.sam2 = 0


st.sidebar.subheader("Give some examples")

sam1 = st.sidebar.button("Example 0.jpg")
sam2 = st.sidebar.button("Example 1.png")
sam3 = st.sidebar.button("Example 2.jpeg")

#############################################################################################
in_file = st.sidebar.file_uploader("Image file uploader:", type=["png", "jpg", "jpeg"])

if in_file is not None:
    st.session_state.sam0=0
    st.session_state.sam1=0
    st.session_state.sam2=0


if sam1:
    st.session_state.in_file0 = open("examples/examples0.jpg", "rb")
    st.session_state.sam0 = 1
    st.session_state.sam1 = 0
    st.session_state.sam2 = 0
    in_file = None
elif sam2:
    st.session_state.in_file1 = open("examples/examples1.jpg", "rb")
    st.session_state.sam0 = 0
    st.session_state.sam1 = 1
    st.session_state.sam2 = 0
    in_file = None
elif sam3:
    st.session_state.in_file2 = open("examples/examples2.png", "rb")
    st.session_state.sam0 = 0
    st.session_state.sam1 = 0
    st.session_state.sam2 = 1
    in_file = None

example=0

st.session_state.md_flag = 1



################################################################################################
pil_image = None

if st.session_state.sam0 == 1:
    pil_image = Image.open(st.session_state.in_file0).convert("RGB")
elif st.session_state.sam1 ==1:
    pil_image = Image.open(st.session_state.in_file1).convert("RGB")
elif st.session_state.sam2 ==1:
    pil_image = Image.open(st.session_state.in_file2).convert("RGB")
elif in_file is not None:
    pil_image = Image.open(in_file).convert("RGB")


col11, col12 = st.sidebar.columns(2)

with col11:
    model_options = ["TokenFD-4096-English-seg", "TokenFD-2048-Bilingual-seg"]
    check_type = st.sidebar.selectbox("Select Model Type", model_options)
    print("check_type",check_type)
with col11:
    text_input = st.text_input("Input the text")
    print("text_input",text_input)
with col11:   
    layout_det = st.button("Run")



if 'vis_list' not in st.session_state:
    st.session_state.vis_list = []


recordmulu = []

if pil_image is None:
    st.stop()




if 'slider_index' not in st.session_state:
    st.session_state.slider_index = 0
# Handle layout detection.  text_input
if layout_det:
    # Reset the slider index to 0 whenever layout_det is triggered
    st.session_state.slider_index = 0

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    checkpoint_vit_english = "TongkunGuan/TokenFD_4096_English_seg"
    checkpoint_vit_bilingual = "TongkunGuan/TokenFD_2048_Bilingual_seg"


    device =torch.device("cuda:0")

    src_iamge_size = pil_image.size
    print('src_iamge_size:{:}'.format(src_iamge_size))

    if check_type == 'R50':
        part1 = torch.load('model/checkpoint_part1.pth', map_location='cpu')['model']
        part2 = torch.load('model/checkpoint_part2.pth', map_location='cpu')['model']
        full_state_dict = {**part1, **part2}
        # """loading model, tokenizer, tok_embeddings """
        tokenizer_path = 'tokenizer_path'
        tokenizer = load_tokenizer(tokenizer_path)
        model = build_model(args).eval()
        model.load_state_dict(full_state_dict)
        model.to(device)
        # """loading image """
        transform = build_transform_R50(normalize_type='imagenet')
        images = pil_image
        pixel_values = torch.stack([transform(images)]).to(device) 
        target_aspect_ratio = (1,1)

    elif check_type == 'TokenFD-4096-English-seg':
        # """loading model, tokenizer, tok_embeddings """
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_vit_english, trust_remote_code=True, use_fast=False)
        model = InternVLChatModel.from_pretrained(checkpoint_vit_english, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,load_in_8bit=False, load_in_4bit=False).eval()
        model = model.to(device)
        # """loading image """
        transform = get_transform(is_train=False, image_size=model.config.force_image_size)
        image = pil_image
        images, target_aspect_ratio = dynamic_preprocess(image, min_num=1, max_num=12,
                                    image_size=model.config.force_image_size, use_thumbnail=model.config.use_thumbnail, return_ratio=True)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values).to(torch.bfloat16).to(model.device)  
    elif check_type == 'TokenFD-2048-Bilingual-seg':
        # """loading model, tokenizer, tok_embeddings """
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_vit_bilingual, trust_remote_code=True, use_fast=False)
        model = InternVLChatModel.from_pretrained(checkpoint_vit_bilingual, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,load_in_8bit=False, load_in_4bit=False).eval()
        model = model.to(device)
        # """loading image """
        transform = get_transform(is_train=False, image_size=model.config.force_image_size)
        image = pil_image
        images, target_aspect_ratio = dynamic_preprocess(image, min_num=1, max_num=12,
                                    image_size=model.config.force_image_size, use_thumbnail=model.config.use_thumbnail, return_ratio=True)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values).to(torch.bfloat16).to(model.device)  

    else:
        raise f'not support type {args.type}'

    # """Load query texts """
    text_input=text_input
    if text_input[0] in '!"#$%&\'()*+,-./0123456789:;<=>?@^_{|}~0123456789':
        input_ids = tokenizer(text_input)['input_ids'][1:]
    else:
        input_ids = tokenizer(' '+text_input)['input_ids'][1:]
    input_ids = torch.Tensor(input_ids).long().to(device)
    with torch.no_grad():
        if check_type == 'R50':
            input_embeds = model.language_embedding(input_ids).clone()
        elif check_type == 'TokenFD-2048-Bilingual-seg':
            input_embeds = model.tok_embeddings(input_ids).clone()
        elif check_type == 'TokenFD-4096-English-seg':
            input_embeds = model.tok_embeddings(input_ids).clone()


    st.session_state.md_flag = 1
    vis_list = []
    text_list = []

    # """Obtaining similarity """
    with torch.no_grad():
        vit_embeds, size1 = model.forward_tokenocr(pixel_values.to(device)) #(vit_batch_size, 16*16, 2048)
    vit_embeds, size2 = post_process(vit_embeds, target_aspect_ratio, check_type)
    vit_embeds = vit_embeds / vit_embeds.norm(dim=-1, keepdim=True)
    input_embedings = input_embeds / input_embeds.norm(dim=-1, keepdim=True)
    similarity = input_embedings @ vit_embeds.t()
    resized_size = size1 if size1 is not None else size2
    attn_map = similarity.reshape(len(input_embedings), resized_size[0], resized_size[1])
    torch.cuda.empty_cache()
    del vit_embeds, input_embedings
    
    # """generate map locally """
    all_bpe_strings = [tokenizer.decode(input_id) for input_id in input_ids]
    vis_list = generate_similiarity_map(images, attn_map, all_bpe_strings,vis_list, target_aspect_ratio, src_iamge_size)

    for vis in vis_list:
        print(vis.shape, vis.dtype)  # Ensure shape is (H, W, 3) and dtype is uint8

    st.session_state.vis_list = vis_list
    st.session_state.bpe = all_bpe_strings

    print("bpe",all_bpe_strings)
    print("ids",input_ids)


# if example==0:
with col1:
    st.subheader(" ")
    st.subheader("Bpe Visualization Demo")
    # Add some empty space above the image
    # Add some space above the image using CSS for more precise control
    st.markdown(
        """
        <div style='height: 38px;'></div>
        """, 
        unsafe_allow_html=True
    )
    st.image(pil_image, caption="Uploaded Image")  



if len(st.session_state.vis_list) > 0:

    with col2:

        if len(st.session_state.vis_list) > 1:
            # Create left and right arrow buttons
            col1, col2, col3 = st.columns([1, 6, 1])
            with col1:
                if st.button('⬅'):
                    st.session_state.slider_index = max(0, st.session_state.slider_index - 1)
            with col3:
                if st.button('⮕'):
                    st.session_state.slider_index = min(len(st.session_state.vis_list) - 1, st.session_state.slider_index + 1)

        if len(st.session_state.vis_list) == 1:
            # Create left and right arrow buttons
            col1, col2, col3 = st.columns([1, 6, 1])
            with col1:
                if st.button('⬅'):
                    st.session_state.slider_index = 1
            with col3:
                if st.button('⮕'):
                    st.session_state.slider_index = 1


        if len(st.session_state.vis_list) > 1:
            # 使用 HTML 进行格式化，使某些文字显示为红色
            # Select the image corresponding to the BPE. 

            st.markdown("""
                <div>
                    <span style="color:red;">If the input text is not included in the image</span>, 
                    the attention map will show a lot of noise 
                    (the actual response value is very low), since we 
                    normalize the attention map according to the relative value.
                </div>
                """, unsafe_allow_html=True
            )

            # 显示滑块，用于选择特定的图像
            index = st.slider(
                'Slide to select an image',
                0,
                len(st.session_state.vis_list) - 1,
                st.session_state.slider_index
            )
            st.session_state.slider_index = index
            # Display the current image based on slider_index


        if len(st.session_state.vis_list) == 1:
            # Display the slider with Markdown to add red color
            st.markdown("""
                <div>
                    <span style="color:red;">If the input text is not included in the image</span>, 
                    the attention map will show a lot of noise 
                    (the actual response value is very low), since we 
                    normalize the attention map according to the relative value.
                </div>
                """, unsafe_allow_html=True
            )

            index = st.slider(
                'Slider Description',
                0,
                1,
                0
            )
            st.session_state.slider_index = 0

        index = st.session_state.slider_index 
        caption_html = f"""
        <div style='color: red; font-weight: bold; font-size: 20px; text-align: center;'>
            BPE: {st.session_state.bpe[index]}
        </div>"""
        st.image(st.session_state.vis_list[index])
        st.markdown(caption_html, unsafe_allow_html=True)
