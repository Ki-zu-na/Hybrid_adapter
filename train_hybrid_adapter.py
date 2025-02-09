import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel, get_cosine_schedule_with_warmup
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

from adapter.habrid_adapter import HybridAdapter
from data_processing.json_adapter_dataset import JSONAdapterDataset
from utils.embedding import get_llama_embedding
from utils.sample import sample_images

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def train(config_path):
    # 从指定路径加载配置文件
    config = OmegaConf.load(config_path)


    # 从配置中读取超参数
    input_dim = config.input_dim
    seq_len = config.seq_len
    mlp_hidden_dim = config.mlp_hidden_dim
    num_transformer_layers = config.num_transformer_layers
    num_attention_heads = config.num_attention_heads
    dropout = config.dropout
    lr = config.lr

    batch_size = config.batch_size
    num_epochs = config.num_epochs
    gradient_accumulation_steps = config.gradient_accumulation_steps
    max_grad_norm = config.max_grad_norm
    sample_every_n_steps = config.sample_every_n_steps
    
    json_data_path = config.json_data_path
    sdxl_model_path = config.sdxl_model_path
    llama_model_path = config.llama_model_path
    output_dir = config.output_dir
    adapter_checkpoint = config.adapter_checkpoint
    use_cross_attn = config.use_cross_attn
    num_workers = config.num_workers



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练 Llama 模型与 tokenizer (仅用于生成 embedding)
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
    llama_model = AutoModel.from_pretrained(llama_model_path).to(device)
    llama_model.requires_grad_(False)  # LLaMA 模型不需要训练
    llama_model.eval()


    # 创建 Hybrid Adapter 模型
    adapter_model = HybridAdapter(
        input_dim=input_dim,
        seq_len=seq_len,
        mlp_hidden_dim=mlp_hidden_dim,
        num_transformer_layers=num_transformer_layers,
        num_attention_heads=num_attention_heads,
        dropout=dropout
    ).to(device)
    
    # 加载预训练的 Adapter 权重 (如果提供)
    if adapter_checkpoint:
        adapter_model.load_state_dict(torch.load(adapter_checkpoint))
        print(f"Loaded adapter checkpoint from: {adapter_checkpoint}")

    # 优化器和损失函数
    optimizer = optim.AdamW(adapter_model.parameters(), lr=lr, weight_decay=1e-2)  # 添加 weight_decay
    criterion = nn.MSELoss()
    # 学习率调度器
    num_training_steps = num_epochs * (len(JSONAdapterDataset(json_data_path, llama_tokenizer, llama_model, sdxl_model_path, device)) // (batch_size*gradient_accumulation_steps))
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_training_steps // 10,  # 10% 的训练步数用于预热
        num_training_steps=num_training_steps,
    )

    # 使用 JSONAdapterDataset 加载 JSON 数据
    dataset = JSONAdapterDataset(
        json_file_path=json_data_path,
        tokenizer=llama_tokenizer,
        llama_model=llama_model,
        sdxl_model_path=sdxl_model_path,
        device=device
    )

    # 定义 worker 初始化函数，在每个 worker 内部加载 SDXL pipeline
    def worker_init_fn(worker_id):
        global worker_sdxl_pipeline
        worker_sdxl_pipeline = StableDiffusionXLPipeline.from_single_file(
            sdxl_model_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        # SDXL 中不需要梯度
        worker_sdxl_pipeline.text_encoder.requires_grad_(False)
        worker_sdxl_pipeline.text_encoder_2.requires_grad_(False)
        worker_sdxl_pipeline.to(device)
        # 卸载无关部分，减少内存占用
        worker_sdxl_pipeline.unet = None
        worker_sdxl_pipeline.vae = None
        if hasattr(worker_sdxl_pipeline, "scheduler"):
            worker_sdxl_pipeline.scheduler = None
        if hasattr(worker_sdxl_pipeline, "safety_checker"):
            worker_sdxl_pipeline.safety_checker = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        worker_init_fn=worker_init_fn
    )

    # AMP 自动混合精度初始化 (仅在CUDA可用时)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    adapter_model.train()
    global_step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for step, (llama_output, (prompt_embeds, pooled_prompt_embeds)) in enumerate(progress):
            llama_output = llama_output.to(device, non_blocking=True)
            prompt_embeds = prompt_embeds.to(device, non_blocking=True) # 这些现在是拼接后的 CLIP embeddings
            pooled_prompt_embeds = pooled_prompt_embeds.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                if use_cross_attn:
                    output_te = adapter_model(llama_output, cross_attn_input=prompt_embeds) # 如果你想用 cross_attn_input，保留它
                else:
                    output_te = adapter_model(llama_output)

                loss = criterion(output_te, prompt_embeds)  # 将 Adapter 输出与拼接后的 prompt_embeds 计算损失
                loss = loss / gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)  # 梯度裁剪前，需要 unscale
                torch.nn.utils.clip_grad_norm_(adapter_model.parameters(), max_grad_norm)  # 梯度裁剪

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * gradient_accumulation_steps * llama_output.size(0)
            progress.set_postfix(loss=loss.item() * gradient_accumulation_steps, lr=lr_scheduler.get_last_lr()[0])

            # 采样
            if global_step % sample_every_n_steps == 0 and global_step != 0:
                # 重新加载完整的SDXL模型用于生成sample
                full_sdxl_pipeline = StableDiffusionXLPipeline.from_single_file(
                    sdxl_model_path,
                    scheduler=DDIMScheduler(
                        beta_start=0.00085,
                        beta_end=0.012,
                        beta_schedule="scaled_linear",
                        num_train_timesteps=1000,
                        clip_sample=False,
                        prediction_type="v_prediction",
                        rescale_betas_zero_snr=True
                    ),
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )

                full_sdxl_pipeline.text_encoder.requires_grad_(False)
                full_sdxl_pipeline.text_encoder_2.requires_grad_(False)
                full_sdxl_pipeline.to(device)

                adapter_model.eval()  # 切换到评估模式
                example_prompt = config["example_prompt"]  # 示例 prompt
                # 生成原始SDXL图像
                sample_images(
                    full_sdxl_pipeline, None, llama_tokenizer, llama_model,
                    example_prompt, device, output_dir, global_step, use_adapter=False
                )
                # 使用 Adapter 辅助生成图像
                sample_images(
                    full_sdxl_pipeline, adapter_model, llama_tokenizer, llama_model,
                    example_prompt, device, output_dir, global_step, use_adapter=True
                )
                adapter_model.train()  # 切换回训练模式

        epoch_loss /= len(dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Global Step: {global_step}")
        
    # 保存最终的 Adapter 模型
    os.makedirs(output_dir, exist_ok=True)
    torch.save(adapter_model.state_dict(), os.path.join(output_dir, "final_adapter.pth"))
    print(f"Final adapter model saved to: {output_dir}/final_adapter.pth")

def parse_args():
    parser = argparse.ArgumentParser(description="Train the Hybrid Adapter")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to the configuration YAML file")
    return parser.parse_args()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    train(args.config)