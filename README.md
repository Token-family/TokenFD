<center>

<h1 style="color: black;">A Token-level Text Image Foundation Model for Document Understanding</h1>


[\[📂 Project Pages\]](https://token-family.github.io/TokenOCR_project/)    [\[📖 Paper\]]()  [\[🤗 Weights\]](https://huggingface.co/TongkunGuan/TokenOCR) [\[🤗 Demo\]](https://huggingface.co/spaces/OpenGVLab/InternVL) [\[🚀 Quick Start\]](#quick-start)  

</center>

<!-- <div align="center">
  <img width="500" alt="image" src="https://cdn-uploads.huggingface.co/production/uploads/64006c09330a45b03605bba3/zJsd2hqd3EevgXo6fNgC-.png">
</div> -->

<center>

## 📖 Table of Contents
- [Introduction](#-introduction)
- [Installation](#%EF%B8%8F-Installation)
- [Quick Start](#-quick-Start)
- [Gradio Demo](#-gradio-Demo)
- [Token Family](#-Token-family)
- [Release Plans](#-Release-Plans)
- [License](#-License)
- [Citation](#-citation)

  
# 📝 Introduction
<h2 style="color: #4CAF50;">Introduction</h2>

</center>

We are excited to announce the release of **`TokenOCR`**, the first token-level visual foundation model specifically tailored for text-image-related tasks, 
designed to support a variety of traditional downstream applications. To facilitate the pretraining of TokenOCR, 
we also devise a high-quality data production pipeline that constructs the first token-level image text dataset, 
**`TokenIT`**, comprising 20 million images and 1.8 billion token-mask pairs. 
Furthermore, leveraging this foundation with exceptional image-as-text capability, 
we seamlessly replace previous VFMs with TokenOCR to construct a document-level MLLM, **`TokenVL`**, for VQA-based document understanding tasks. 

In summary:

(1) The first token-level image text dataset (TokenIT) is proposed;

(2) The first token-level text image foundation model, TokenOCR, is proposed to support downstream tasks.

(3) The image-as-text semantic capability inspires us to develop TokenVL, a VQA-based MLLM tailored for document perception, understanding, and reasoning.
<center>

## 🛠️ Installation
<h2 style="color: #4CAF50;">Installation</h2>

</center>

```
conda create -n tokenocr python=3.9
conda activate tokenocr
pip install -r requirements.txt
```
Install flash-attn==2.3.6 (optional):
```
pip install flash-attn==2.3.6 --no-build-isolation
```
Alternatively you can compile from source:
```
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.3.6
python setup.py install
```
If you don't use flash-attn, please modify the configs of [weights](https://huggingface.co/TongkunGuan/TokenOCR/tree/main), referring to [this](https://github.com/OpenGVLab/InternVL/issues/163#issuecomment-2114083407)

<center>

## 🚀 Quick Start

> \[!Warning\]
> 🚨 Note: In our experience, the `TokenOCR-2048-Bilingual` series is better suited for building MLLMs than the `-seg` version.

```python
import os
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from internvl.model.internvl_chat import InternVLChatConfig, InternVisionModel
from utils import  post_process, generate_similiarity_map, load_model, load_image

checkpoint = 'xxx/checkpoint-370000'
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
model = model.cuda()
tok_embeddings = tok_embeddings.cuda()

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
generate_similiarity_map(images, attn_map, target_aspect_ratio, all_bpe_strings, out_dir)
```
## ✨ Gradio Demo
<h2 style="color: #4CAF50;">Gradio Demo</h2>

<center>
To run gradio demo, simply run:

```
streamlit run gtk.py --server.port 8420
```

Then a simple Web-UI to interactive:
<div align="center">
  <img width="1500" alt="image" src="https://github.com/user-attachments/assets/5d427eeb-a50e-4bd6-9239-8c2a9d2b072f">
</div>
</center>

## 🏠 Token Family (TokenIT, TokenOCR, TokenVL)
<h2 style="color: #4CAF50;">Token Family (TokenIT, TokenOCR, TokenVL)</h2>

</center>


<details><summary>TokenIT</summary>
<h2 style="color: #4CAF50;">TokenIT</h2>

In the following picture, we provide an overview of the self-constructed token-level **TokenIT** dataset, comprising 20 million images and 1.8 billion
text-mask pairs. 

As depicted in Figure 2 (a), each sample in this dataset includes a raw image, a mask image, and a JSON file. 
The JSON file provides the question-answer pairs and several BPE tokens randomly selected from the answer, along with 
the ordinal number of each BPE token in the answer and its corresponding pixel value on the mask image. Consequently,
**each BPE token corresponds one-to-one with a pixel-level mask**. 
The data ratios are summarized in Figure 2 (b). Figure 2 (c) and (d) further provide the number distribution 
of tokens per image type and a word cloud of the top 100 tokens, respectively.

<div align="center">
  <img width="1000" alt="image" src="https://cdn-uploads.huggingface.co/production/uploads/650d4a36cbd0c7d550d3b41b/WcQwU3-xjyT5Vm-pZhACo.png">
</div>

<!-- ![image/png](https://cdn-uploads.huggingface.co/production/uploads/650d4a36cbd0c7d550d3b41b/WcQwU3-xjyT5Vm-pZhACo.png) -->

The comparisons with other visual foundation models:

| VFM                | Granularity | Dataset  | #Image | #Pairs |
|:-------------------|:------------|:---------|:------:|:------:|
| [CLIP](https://github.com/openai/CLIP) | image-level | WIT400M  | 400M   | 0.4B   |
| [DINO](https://github.com/facebookresearch/dino) | image-level | ImageNet | 14M    | -      |
| [SAM](https://github.com/facebookresearch/SAM)  | pixel-level | SA1B     | 11M    | 1.1B   |
| **TokenOCR**           | **token-level** | **TokenIT**  | **20M**    | **1.8B**   |

</details>


<details><summary>TokenOCR</summary>
<h2 style="color: #4CAF50;">TokenOCR</h2>

### Model Architecture

An overview of the proposed TokenOCR, where the token-level image features and token-level language
features are aligned within the same semantic space. This “image-as-text” alignment seamlessly facilitates user-interactive
applications, including text segmentation, retrieval, and visual question answering.

<div align="center">
  <img width="1000" alt="image" src="https://cdn-uploads.huggingface.co/production/uploads/650d4a36cbd0c7d550d3b41b/QTsvWxFJFTnISdhvbfZhD.png">
</div>

### Model Cards

In the following table, we provide all models [🤗 link](https://huggingface.co/TongkunGuan/TokenOCR/tree/main) of the TokenOCR series.

|        Model Name         |                                Description                                |
| :-----------------------: | :-------------------------------------------------------------------: |
| TokenOCR-4096-English | feature dimension is 4096; support interactive with English texts.|
|  TokenOCR-4096-Chinese  |  feature dimension is 4096; support interactive with Chinese texts.  |
|  TokenOCR-2048-Bilingual  |  feature dimension is 4096; support interactive with English and Chinese texts. |
| TokenOCR-4096-English-seg |  On `TokenOCR-4096-English`, background noise is filtered out. You can use prompt ' ' to get a highlight background. |

### Evaluation on Vision Capability

We present a comprehensive evaluation of the vision encoder’s performance across various domains and tasks. 
The evaluation is divided into two key categories:

(1) text retrial; 
(2) image segmentation;
(3) visual question answering;

This approach allows us to assess the representation quality of TokenOCR. 
Please refer to our technical report for more details.

#### text retrial

<div align="left">
  <img width="500" alt="image" src="https://cdn-uploads.huggingface.co/production/uploads/650d4a36cbd0c7d550d3b41b/b2b2g23o9GMmPe1PiCn0f.png">
</div>


<!-- ![image/png](https://cdn-uploads.huggingface.co/production/uploads/650d4a36cbd0c7d550d3b41b/b2b2g23o9GMmPe1PiCn0f.png) -->

#### image segmentation

<div align="left">
  <img width="500" alt="image" src="https://cdn-uploads.huggingface.co/production/uploads/650d4a36cbd0c7d550d3b41b/C15-Ica6XVfX6y_MgiVds.png">
</div>

<!-- ![image/png](https://cdn-uploads.huggingface.co/production/uploads/650d4a36cbd0c7d550d3b41b/C15-Ica6XVfX6y_MgiVds.png) -->

#### visual question answering

<div align="left">
  <img width="500" alt="image" src="https://cdn-uploads.huggingface.co/production/uploads/650d4a36cbd0c7d550d3b41b/IbLZ0CxCxDkTaHAMe7M0Q.png">
</div>

<!-- ![image/png](https://cdn-uploads.huggingface.co/production/uploads/650d4a36cbd0c7d550d3b41b/IbLZ0CxCxDkTaHAMe7M0Q.png)
 -->
</details>

<details><summary>TokenVL</summary>
<h2 style="color: #4CAF50;">TokenVL </h2>

we employ the TokenOCR as the visual foundation model and further develop an MLLM, named TokenVL, tailored for document understanding. 
Following the previous training paradigm, TokenVL also includes two stages: 

**Stage 1: LLM-guided Token Alignment Training for text parsing tasks.**

<div align="center">
  <img width="500" alt="image" src="https://cdn-uploads.huggingface.co/production/uploads/650d4a36cbd0c7d550d3b41b/gDr1fQg7I1nTIsiRWNHTr.png">
</div>

The framework of LLM-guided Token Alignment Training. Existing MLLMs primarily enhance spatial-wise text perception capabilities by integrating localization prompts to predict coordinates. However, this implicit
method makes it difficult for these models to have a precise understanding. 
In contrast, the proposed token alignment uses BPE token masks to directly and explicitly align text with corresponding pixels in the input image, enhancing the MLLM’s localization awareness.

**Stage 2: Supervised Instruction Tuning for VQA tasks.**

During the Supervised Instruction Tuning stage, we cancel the token alignment branch as answers may not appear in the image for some reasoning tasks 
(e.g., How much taller is the red bar compared to the green bar?). This also ensures no computational overhead during inference to improve the document understanding capability. Finally, we inherit the
remaining weights from the LLM-guided Token Alignment and unfreeze all parameters to facilitate comprehensive parameter updates.

### OCRBench Results

<div align="center">
  <img width="1300" alt="image" src="https://cdn-uploads.huggingface.co/production/uploads/650d4a36cbd0c7d550d3b41b/DZej5Ogpho3wpZC4KVAMO.png">
</div>

### Document Understanding Results

<div align="center">
  <img width="1300" alt="image" src="https://cdn-uploads.huggingface.co/production/uploads/650d4a36cbd0c7d550d3b41b/Msfs1YkDQHq2-djhm6QqD.png">
</div>
</details>

## 🤚 Release Plans

- [✅] Inference code and weights for TokenOCR
- [x] TokenIT data and script
- [x] Data for the Pre-training and Fine-tuning of TokenVL
- [x] Code & model checkpoint for TokenVL
- [x] Other code and resources
- [x] Code for pre-training and evaluation on ScreenSpot
- [x] Code for collecting pre-training data

## 🏛 License

This project is released under the MIT License.

## 📎 Citation

If you find this project useful in your research, please consider citing:

```BibTeX

```
