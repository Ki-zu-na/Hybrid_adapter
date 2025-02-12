import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import PIL
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel, get_cosine_schedule_with_warmup
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

from adapter.habrid_adapter import HybridAdapter
from data_processing.json_adapter_dataset import JSONAdapterDataset
from utils.embedding import get_llama_embedding


def sample_images(sdxl_pipeline, adapter_model, llama_tokenizer, llama_model, prompt, device, output_dir, step, use_adapter=True):
    """
    生成图像样本，用于对比原始 SDXL 和使用 Adapter 后的效果。

    Args:
        sdxl_pipeline:  已加载的 SDXL pipeline.
        adapter_model:  Hybrid Adapter 模型 (可能为 None).
        llama_tokenizer: LLaMA tokenizer.
        llama_model: LLaMA 模型.
        prompt: 用于生成图像的文本提示.
        device:  torch.device.
        output_dir: 保存图像的目录.
        step: 当前训练步数 (用于文件名).
        use_adapter: 是否使用 Adapter (True: 使用, False: 使用原始 SDXL).
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"sample_{'adapter' if use_adapter else 'original'}_step_{step}.png"
    filepath = os.path.join(output_dir, filename)

    if use_adapter:
        # 1. 获取 LLaMA embedding
        llama_emb = get_llama_embedding(prompt, llama_tokenizer, llama_model, device)  # [1, hidden_dim]
        llama_emb = llama_emb.unsqueeze(0) # [batch_size, hidden_dim]

        # 2. 通过 Adapter 转换，获取 prompt_embeds 和 pooled_prompt_embeds
        adapter_model.eval() # 确保在 eval 模式
        with torch.no_grad():
            adapter_prompt_embeds, adapter_pooled_prompt_embeds = adapter_model(llama_emb)  # 返回两个 embeddings
            adapter_prompt_embeds = adapter_prompt_embeds.to(dtype=sdxl_pipeline.text_encoder_2.dtype)
            adapter_pooled_prompt_embeds = adapter_pooled_prompt_embeds.to(dtype=sdxl_pipeline.text_encoder_2.dtype)


        # 3. 使用 Adapter 输出的 embeddings 生成图像
        image = sdxl_pipeline(
            prompt_embeds=adapter_prompt_embeds,  # 使用 adapter_prompt_embeds
            pooled_prompt_embeds=adapter_pooled_prompt_embeds, # 使用 adapter_pooled_prompt_embeds
            output_type="pil"
        ).images[0]
    else:
       # 使用原始 SDXL 生成图像 (保持不变)
        image = sdxl_pipeline(prompt=prompt, output_type="pil").images[0]


    image.save(filepath)
    print(f"Sample image saved to: {filepath}")