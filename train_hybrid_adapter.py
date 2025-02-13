import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel,  get_cosine_with_hard_restarts_schedule_with_warmup
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

import wandb

from adapter.habrid_adapter import HybridAdapter
from data_processing.json_adapter_dataset import JSONAdapterDataset, custom_collate_fn
from utils.embedding import get_llama_embedding
from utils.sample import sample_images


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def train(config_path):
    # 从指定路径加载配置文件
    config = OmegaConf.load(config_path)

    # 初始化 wandb (确保在配置文件中添加 wandb_project 参数)
    wandb.init(
        project=config.get("wandb_project", "default_project"),
        config=OmegaConf.to_container(config, resolve=True)
    )

    # 从配置中读取超参数
    input_dim = config.get("input_dim", 2048)
    seq_len = config.get("seq_len", 512)
    clip_l_dim = config.get("clip_l_dim", 768)
    clip_g_dim = config.get("clip_g_dim", 1280)
    mlp_hidden_dim = config.get("mlp_hidden_dim", 4096)
    num_transformer_layers = config.get("num_transformer_layers", 2)
    num_attention_heads = config.get("num_attention_heads", 8)
    dropout = config.get("dropout", 0.1)
    lr = config.get("lr", 5e-5)
    v_parameterization = config.get("v_parameterization", False)

    batch_size = config.get("batch_size", 16)
    num_epochs = config.get("num_epochs", 10)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 2)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    sample_every_n_steps = config.get("sample_every_n_steps", 1000)

    checkpoint_save_steps = config.get("checkpoint_save_steps", 1000)


    json_data_path = config.json_data_path
    sdxl_model_path = config.sdxl_model_path
    llama_model_path = config.llama_model_path
    output_dir = config.get("output_dir", "output")
    adapter_checkpoint = config.get("adapter_checkpoint", None)
    use_cross_attn = config.get("use_cross_attn", True)
    num_workers = config.get("num_workers", 16)


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
        dropout=dropout,
        clip_l_dim=clip_l_dim, # 添加 clip_l_dim 参数
        clip_g_dim=clip_g_dim # 添加 clip_g_dim 参数
    ).to(device)
    
    # 加载预训练的 Adapter 权重 (如果提供)
    if adapter_checkpoint:
        adapter_model.load_state_dict(torch.load(adapter_checkpoint))
        print(f"Loaded adapter checkpoint from: {adapter_checkpoint}")

    # 优化器和损失函数
    optimizer = optim.AdamW(adapter_model.parameters(), lr=lr, weight_decay=1e-2)  # 添加 weight_decay
    criterion_prompt = nn.MSELoss()  # prompt_embeds 的损失
    criterion_pooled = nn.MSELoss()  # pooled_prompt_embeds 的损失
    # 学习率调度器
    num_training_steps = num_epochs * (len(JSONAdapterDataset(json_data_path, llama_tokenizer, llama_model, sdxl_model_path, device)) // (batch_size * gradient_accumulation_steps))
    lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,  # 100的训练步数用于预热
        num_training_steps=num_training_steps,
        num_cycles=20
    )

    # 使用 JSONAdapterDataset 加载 JSON 数据
    dataset = JSONAdapterDataset(
        json_file_path=json_data_path,
        tokenizer=llama_tokenizer,
        llama_model=llama_model,
        sdxl_model_path=sdxl_model_path,

        device=device
    )
    
    def generate_sample():
        full_sdxl_pipeline = StableDiffusionXLPipeline.from_single_file(
        sdxl_model_path,
        scheduler=DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            prediction_type="v_prediction" if v_parameterization else "epsilon",
            rescale_betas_zero_snr=True if v_parameterization else False
        ),
        torch_dtype=torch.float16,
        use_safetensors=True
        )

        full_sdxl_pipeline.text_encoder.requires_grad_(False)
        full_sdxl_pipeline.text_encoder_2.requires_grad_(False)
        full_sdxl_pipeline.to(device)

        adapter_model.eval()  # 切换到评估模式
        example_prompt = config["example_prompt"]  # 示例 prompt

        # 生成原始SDXL图像，并假设 sample_images 保存图片文件至 output_dir，如 sample_original_step_{global_step}.png
        sample_images(
            full_sdxl_pipeline, None, llama_tokenizer, llama_model,
            example_prompt, device, output_dir, global_step, use_adapter=False
        )
        # 使用 Adapter 辅助生成图像，并假设保存为 sample_adapter_step_{global_step}.png
        hidden_size = full_sdxl_pipeline.text_encoder_2.config.hidden_size
        pooled_prompt_embeds = torch.randn(1, hidden_size, device=device, dtype=full_sdxl_pipeline.text_encoder_2.dtype)
        sample_images(
            full_sdxl_pipeline, adapter_model, llama_tokenizer, llama_model,
            example_prompt, device, output_dir, global_step, use_adapter=True
        )
        
        try:
            from PIL import Image
            original_sample_path = os.path.join(output_dir, f"sample_original_step_{global_step}.png")
            adapter_sample_path = os.path.join(output_dir, f"sample_adapter_step_{global_step}.png")
            original_img = Image.open(original_sample_path)
            adapter_img = Image.open(adapter_sample_path)

            # 使用 wandb 记录采样图像
            wandb.log({
                "sample_original": wandb.Image(original_img, caption=f"Step {global_step} Original"),
                "sample_adapter": wandb.Image(adapter_img, caption=f"Step {global_step} Adapter")
            }, step=global_step)
        except Exception as e:
            print(f"Error logging sample images to wandb at step {global_step}: {e}")

        adapter_model.train()  # 切换回训练模式
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
        worker_init_fn=worker_init_fn,
        collate_fn=custom_collate_fn
    )

    # AMP 自动混合精度初始化 (仅在CUDA可用时)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    adapter_model.train()
    global_step = 0
    if config.get("sample_at_start", False):
        generate_sample()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for step, (llama_output, (prompt_embeds, pooled_prompt_embeds)) in enumerate(progress):
            llama_output = llama_output.to(device, non_blocking=True)
            prompt_embeds = prompt_embeds.to(device, non_blocking=True)  # 这些现在是拼接后的 CLIP embeddings
            pooled_prompt_embeds = pooled_prompt_embeds.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                if use_cross_attn:
                    adapter_prompt_embeds, adapter_pooled_prompt_embeds = adapter_model(llama_output, cross_attn_input=prompt_embeds)  # 获取两个输出
                    output_te_prompt = adapter_prompt_embeds # 用于计算 prompt_loss
                    output_te_pooled = adapter_pooled_prompt_embeds # 用于计算 pooled_loss
                else:
                    adapter_prompt_embeds, adapter_pooled_prompt_embeds = adapter_model(llama_output) # 获取两个输出
                    output_te_prompt = adapter_prompt_embeds # 用于计算 prompt_loss
                    output_te_pooled = adapter_pooled_prompt_embeds # 用于计算 pooled_loss

                # 分别计算 prompt_embeds 和 pooled_prompt_embeds 的损失
                prompt_loss = criterion_prompt(output_te_prompt, prompt_embeds)
                pooled_loss = criterion_pooled(output_te_pooled, pooled_prompt_embeds)

                # 将两个损失加权求和，得到总损失 (你可以调整权重)
                total_loss = 5 * prompt_loss + pooled_loss #  简单的加和，你可以尝试加权，例如： total_loss = prompt_loss + 0.5 * pooled_loss
                loss = total_loss / gradient_accumulation_steps # 梯度累积需要除以步数

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

                # 使用 wandb 记录训练的 loss 和 lr (修改 wandb 日志)
                wandb.log({
                    "train_prompt_loss": prompt_loss.item() * gradient_accumulation_steps, # 记录 prompt_loss
                    "train_pooled_loss": pooled_loss.item() * gradient_accumulation_steps, # 记录 pooled_loss
                    "train_total_loss": total_loss.item() * gradient_accumulation_steps, # 记录 total_loss
                    "lr": lr_scheduler.get_last_lr()[0],
                    "global_step": global_step,
                    "epoch": epoch + 1
                }, step=global_step)

            epoch_loss += total_loss.item() * gradient_accumulation_steps * llama_output.size(0) # epoch_loss 也要累加 total_loss
            progress.set_postfix(total_loss=total_loss.item() * gradient_accumulation_steps, lr=lr_scheduler.get_last_lr()[0]) # progress bar 显示 total_loss

            # 采样，并记录 sample 到 wandb
            if global_step % sample_every_n_steps == 0 :
                # 重新加载完整的SDXL模型用于生成 sample
                generate_sample()


            # 新增：每隔固定步数储存一次模型
            if global_step % checkpoint_save_steps == 0 and global_step != 0:
                checkpoint_path = os.path.join(output_dir, f'adapter_checkpoint_step_{global_step}.pth')
                torch.save(adapter_model.state_dict(), checkpoint_path)
                print(f"Saved adapter checkpoint at step {global_step} to {checkpoint_path}")

        epoch_loss /= len(dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Global Step: {global_step}")
        
    # 保存最终的 Adapter 模型
    os.makedirs(output_dir, exist_ok=True)
    torch.save(adapter_model.state_dict(), os.path.join(output_dir, "final_adapter.pth"))
    print(f"Final adapter model saved to: {output_dir}/final_adapter.pth")
    wandb.save(os.path.join(output_dir, "final_adapter.pth"))


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Hybrid Adapter")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to the configuration YAML file")
    return parser.parse_args()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    train(args.config)