import os
import json
import random
import torch
from torch.utils.data import Dataset
from diffusers import StableDiffusionXLPipeline
from utils.embedding import get_llama_embedding


class JSONAdapterDataset(Dataset):
    """
    使用 JSON 数据来动态生成 Llama 与 SDXL CLIP 的 embedding.
    
    JSON 中的每一项应包含如下字段（部分为可选字段）：
        - tag_string_artist: 艺术家标签
        - tag_string_character: 人物标签
        - tag_string_copyright: 版权标签
        - tag_string_general: 通用描述（以逗号分隔）
        - rating: 内容评级 ('e', 's', 'q', 'g')
        - aes_rating: AES 评级信息
        - tag_string_meta: 元信息标签
        - regular_summary: 常规描述
        - brief_summary: 简短描述
    
    Args:
        json_file_path (str): JSON 文件的路径.
        tokenizer: 用于 Llama 模型的 tokenizer.
        llama_model: 预训练的 Llama 模型.
        sdxl_model_path: SDXL pipeline 的路径.
        device: torch.device 对象.
        drop_artist_prob (float): 删除艺术家标签的概率.
        caption_nl_prob (float): 使用自然语言描述（regular_summary）的概率.
        style_mix_prob (float): 使用 brief_summary 的概率.
        drop_all_fixed_prob (float): 删除所有固定标签（艺术家、人物、版权）的概率.
        drop_all_flex_prob (float): 删除所有灵活标签的概率.
        dropout_rate (float): 对灵活标签随机丢弃的概率.
        shuffle_caption (bool): 是否随机打乱标签顺序.
    """
    def __init__(self, json_file_path, tokenizer, llama_model, sdxl_model_path, device,
                 drop_artist_prob=0.3, caption_nl_prob=0.2, style_mix_prob=0.1,
                 drop_all_fixed_prob=0.1, drop_all_flex_prob=0.1, dropout_rate=0.1,
                 shuffle_caption=True):
        super().__init__()
        with open(json_file_path, "r", encoding="utf-8") as f:
            self.json_data = json.load(f)
        self.keys = list(self.json_data.keys())
        self.tokenizer = tokenizer
        self.llama_model = llama_model
        self.sdxl_model_path = sdxl_model_path  # 保存路径，用于延迟加载
        self.device = device

        # 不在此处加载 SDXL pipeline，避免 pickle 问题
        self.sdxl_model = None

        # 参数设置
        self.drop_artist_prob = drop_artist_prob
        self.caption_nl_prob = caption_nl_prob
        self.style_mix_prob = style_mix_prob
        self.drop_all_fixed_prob = drop_all_fixed_prob
        self.drop_all_flex_prob = drop_all_flex_prob
        self.dropout_rate = dropout_rate
        self.shuffle_caption = shuffle_caption

    @staticmethod
    def dropout_tags(tags, dropout_rate):
        """
        随机丢弃部分标签以增强灵活性.
        """
        return [tag for tag in tags if random.random() > dropout_rate]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # 延迟加载 SDXL pipeline：优先使用 worker_init_fn 中初始化的全局变量
        if self.sdxl_model is None:
            try:
                global worker_sdxl_pipeline
                self.sdxl_model = worker_sdxl_pipeline
            except (NameError, AttributeError):
                # 如果不在 worker 中，则回退到直接初始化
                self.sdxl_model = StableDiffusionXLPipeline.from_single_file(
                    self.sdxl_model_path,
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )
                self.sdxl_model.text_encoder.requires_grad_(False)
                self.sdxl_model.text_encoder_2.requires_grad_(False)
                self.sdxl_model.to(self.device)
                self.sdxl_model.unet = None
                self.sdxl_model.vae = None
                if hasattr(self.sdxl_model, "scheduler"):
                    self.sdxl_model.scheduler = None
                if hasattr(self.sdxl_model, "safety_checker"):
                    self.sdxl_model.safety_checker = None

        item_key = self.keys[idx]
        extras = self.json_data[item_key]

        # 固定标签处理
        fixed_tags = []
        drop_artist = random.random() < self.drop_artist_prob
        if 'tag_string_artist' in extras and extras['tag_string_artist'] and not drop_artist:
            fixed_tags.append(extras['tag_string_artist'])
        if 'tag_string_character' in extras and extras['tag_string_character']:
            fixed_tags.append(extras['tag_string_character'])
        if 'tag_string_copyright' in extras and extras['tag_string_copyright']:
            fixed_tags.append(extras['tag_string_copyright'])

        # 灵活标签处理
        flex_tags = []
        caption_nl = random.random() < self.caption_nl_prob
        style_mix = random.random() < self.style_mix_prob
        if 'tag_string_general' in extras and extras['tag_string_general'] and not caption_nl:
            # 按逗号分割并去除多余空格
            flex_tags = [tag.strip() for tag in extras['tag_string_general'].split(",") if tag.strip()]
            if 'rating' in extras and extras['rating']:
                if extras['rating'] == 'e':
                    flex_tags.append('explicit')
                elif extras['rating'] == 's':
                    flex_tags.append('sensitive')
                elif extras['rating'] == 'q':
                    flex_tags.append('questionable')
                elif extras['rating'] == 'g':
                    flex_tags.append('safe')
            if 'aes_rating' in extras and extras['aes_rating']:
                flex_tags.append(extras['aes_rating'])
            if 'tag_string_meta' in extras and extras['tag_string_meta']:
                flex_tags.append(extras['tag_string_meta'])
        elif 'regular_summary' in extras and extras['regular_summary'] and caption_nl:
            flex_tags.append(extras['regular_summary'])
        elif 'brief_summary' in extras and extras['brief_summary'] and style_mix:
            flex_tags.append(extras['brief_summary'])

        # 决定是否丢弃所有固定/灵活标签
        drop_all_fixed = random.random() < self.drop_all_fixed_prob
        drop_all_flex = random.random() < self.drop_all_flex_prob

        if drop_all_fixed:
            fixed_tags = []
        if drop_all_flex:
            flex_tags = []
        else:
            flex_tags = self.dropout_tags(flex_tags, self.dropout_rate)

        # 可选：随机打乱标签顺序
        if self.shuffle_caption:
            random.shuffle(fixed_tags)
            random.shuffle(flex_tags)

        new_prompt = ", ".join(fixed_tags + flex_tags)
        if not new_prompt:
            new_prompt = ""

        # 计算 Llama embedding
        llama_emb = get_llama_embedding(new_prompt, self.tokenizer, self.llama_model, self.device)

        # 使用官方方式计算 SDXL 文本嵌入 (同时获取 text_encoder 和 text_encoder_2 的 embedding)
        with torch.no_grad():
            tokenizer_l = self.sdxl_model.tokenizer
            tokenizer_g = self.sdxl_model.tokenizer_2
            text_encoder_l = self.sdxl_model.text_encoder
            text_encoder_g = self.sdxl_model.text_encoder_2

            max_length_l = tokenizer_l.model_max_length - 2
            max_length_g = tokenizer_g.model_max_length - 2

            # 分割 prompt 成段落，保证每段不超过 tokenizer 的最大长度
            prompt_chunks_l = self._chunk_prompt(new_prompt, tokenizer_l, max_length_l)
            prompt_chunks_g = self._chunk_prompt(new_prompt, tokenizer_g, max_length_g)

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
                encoded_input_l = {k: v.to(self.device) for k, v in encoded_input_l.items()}
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
                encoded_input_g = {k: v.to(self.device) for k, v in encoded_input_g.items()}
                text_outputs_g = text_encoder_g(**encoded_input_g, output_hidden_states=True)
                prompt_embeds_clip_g = text_outputs_g.hidden_states[-2]
                pooled_prompt_embeds_clip_g = text_outputs_g[0]

                prompt_embeds_clip_g_list.append(prompt_embeds_clip_g)
                pooled_prompt_embeds_clip_g_list.append(pooled_prompt_embeds_clip_g)

            # 拼接所有段落的 prompt embeddings (在序列长度维度上)
            concatenated_prompt_embeds_l = torch.cat(prompt_embeds_clip_l_list, dim=1) # [1, seq_len_total_l, hidden_dim_l]
            concatenated_prompt_embeds_g = torch.cat(prompt_embeds_clip_g_list, dim=1) # [1, seq_len_total_g, hidden_dim_g]

            # 简单平均池化所有段落的 pooled prompt embeddings
            pooled_prompt_embeds_clip_l = torch.cat(pooled_prompt_embeds_clip_l_list, dim=1).mean(dim=1) # [1, hidden_dim_l]
            pooled_prompt_embeds_clip_g = torch.cat(pooled_prompt_embeds_clip_g_list, dim=1).mean(dim=1) # [1, hidden_dim_g]

            # 拼接 CLIP-L 和 CLIP-G 的 prompt embeddings
            concatenated_prompt_embeds = torch.cat((concatenated_prompt_embeds_l.squeeze(0), concatenated_prompt_embeds_g.squeeze(0)), dim=-1) # [seq_len_total, hidden_dim_l + hidden_dim_g]
            #concatenated_pooled_prompt_embeds = torch.cat((pooled_prompt_embeds_clip_l.reshape(-1), pooled_prompt_embeds_clip_g.reshape(-1)), dim=-1) # [hidden_dim_l + hidden_dim_g]


        # 计算 Llama embedding (保持不变)
        llama_emb = get_llama_embedding(new_prompt, self.tokenizer, self.llama_model, self.device)

        return llama_emb, (concatenated_prompt_embeds, pooled_prompt_embeds_clip_g) # 返回拼接后的 embedding

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
