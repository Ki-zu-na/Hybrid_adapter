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
        tokenizer_l = sdxl_pipeline.tokenizer
        tokenizer_g = sdxl_pipeline.tokenizer_2
        text_encoder_l = sdxl_pipeline.text_encoder
        text_encoder_g = sdxl_pipeline.text_encoder_2

        max_length_l = tokenizer_l.model_max_length
        max_length_g = tokenizer_g.model_max_length

        # 分割 prompt 成段落，保证每段不超过 tokenizer 的最大长度
        prompt_chunks_l = _chunk_prompt(prompt, tokenizer_l, max_length_l)
        prompt_chunks_g = _chunk_prompt(prompt, tokenizer_g, max_length_g)

        prompt_embeds_clip_l_list = []
        pooled_prompt_embeds_clip_l_list = []
        prompt_embeds_clip_g_list = []
        pooled_prompt_embeds_clip_g_list = []

        # 循环处理每个 prompt 段落 (CLIP ViT-L/14)
        for chunk in prompt_chunks_l:
            encoded_input_l = tokenizer_l(
                chunk,
                padding="max_length",
                max_length=max_length_l,
                truncation=False, #  重要：这里设置为 False，不要截断，我们已经分段了
                return_tensors="pt",
            )
            encoded_input_l = {k: v.to(device) for k, v in encoded_input_l.items()}
            text_outputs_l = text_encoder_l(**encoded_input_l, output_hidden_states=True)
            prompt_embeds_clip_l = text_outputs_l.hidden_states[-2]
            pooled_prompt_embeds_clip_l = text_outputs_l[0]

            prompt_embeds_clip_l_list.append(prompt_embeds_clip_l)
            pooled_prompt_embeds_clip_l_list.append(pooled_prompt_embeds_clip_l)

        # 循环处理每个 prompt 段落 (CLIP ViT-bigG/14)
        for chunk in prompt_chunks_g:
            encoded_input_g = tokenizer_g(
                chunk,
                padding="max_length",
                max_length=max_length_g,
                truncation=False, # 重要：这里设置为 False，不要截断，我们已经分段了
                return_tensors="pt",
            )
            encoded_input_g = {k: v.to(device) for k, v in encoded_input_g.items()}
            text_outputs_g = text_encoder_g(**encoded_input_g, output_hidden_states=True)
            prompt_embeds_clip_g = text_outputs_g.hidden_states[-2]
            pooled_prompt_embeds_clip_g = text_outputs_g[0]

            prompt_embeds_clip_g_list.append(prompt_embeds_clip_g)
            pooled_prompt_embeds_clip_g_list.append(pooled_prompt_embeds_clip_g)

        # 拼接所有段落的 prompt embeddings (在序列长度维度上)
        concatenated_prompt_embeds_l = torch.cat(prompt_embeds_clip_l_list, dim=1) # [1, seq_len_total_l, hidden_dim_l]
        concatenated_prompt_embeds_g = torch.cat(prompt_embeds_clip_g_list, dim=1) # [1, seq_len_total_g, hidden_dim_g]
        concatenated_prompt_embeds = torch.cat((concatenated_prompt_embeds_l.squeeze(0), concatenated_prompt_embeds_g.squeeze(0)), dim=-1) # [seq_len_total, hidden_dim_l + hidden_dim_g]

        image = sdxl_pipeline(
            prompt_embeds=concatenated_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds_clip_g,
            output_type="pil").images[0]


    image.save(filepath)
    print(f"Sample image saved to: {filepath}")

    def _chunk_prompt(self, prompt_text, tokenizer, max_length):
        """
        将长文本 prompt 分割成多个段落，每个段落长度不超过 max_length。
        """
        tokens = tokenizer.tokenize(prompt_text)
        chunks = []
        current_chunk_tokens = []
        for token in tokens:
            current_chunk_tokens.append(token)
            if len(current_chunk_tokens) >= max_length:
                chunks.append(tokenizer.convert_tokens_to_string(current_chunk_tokens))
                current_chunk_tokens = []
        if current_chunk_tokens: # 处理最后剩余的 chunk
            chunks.append(tokenizer.convert_tokens_to_string(current_chunk_tokens))
        return chunks
