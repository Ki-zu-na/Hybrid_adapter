import torch

# 由于直接使用 sdxl_model.get_text_embeddings, 这个函数不再需要
# def get_clip_embedding(text, model, device):
#     """
#     Compute the SDXL CLIP text embedding.  直接使用 sdxl_model.get_text_embeddings
#     """
#     pass

def get_llama_embedding(text, tokenizer, model, device):
    """
    Compute the Llama embedding for the provided text.

    Args:
        text (str): Input text.
        tokenizer: A tokenizer from the Transformers library.
        model: A pre-trained Llama model.
        device: torch.device object.

    Returns:
        torch.Tensor: The computed text embedding.
    """
    # Tokenize the input text
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    with torch.no_grad():
        # Forward pass得到输出
        model_output = model(**encoded_input)
    
    # 取最后隐藏状态并平均池化生成 embedding
    last_hidden_state = model_output.last_hidden_state  # [batch_size, seq_length, hidden_dim]
    embedding = last_hidden_state.mean(dim=1)            # [batch_size, hidden_dim]
    
    return embedding.squeeze(0)