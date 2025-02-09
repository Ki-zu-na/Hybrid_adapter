import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlockWithCrossAttn(nn.Module):
    """
    Transformer Block with optional cross attention.
    If cross_attn_input is provided, applies cross attention after self-attention.
    Uses pre-LN structure.
    """
    def __init__(self, input_dim: int, num_heads: int, hidden_dim: int, dropout: float, cross_attn_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.cross_attn_dim = cross_attn_dim

        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        # Cross-attention layer
        self.cross_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True, kdim=cross_attn_dim, vdim=cross_attn_dim)

        # 手动创建 q, k, v 投影权重
        self.q_proj_weight = nn.Parameter(torch.empty((input_dim, input_dim)))
        self.k_proj_weight = nn.Parameter(torch.empty((input_dim, cross_attn_dim)))
        self.v_proj_weight = nn.Parameter(torch.empty((input_dim, cross_attn_dim)))

        # 将手动创建的权重赋值给 cross_attn
        self.cross_attn.q_proj_weight = self.q_proj_weight
        self.cross_attn.k_proj_weight = self.k_proj_weight
        self.cross_attn.v_proj_weight = self.v_proj_weight


        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

        # Pre-LN LayerNorms
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cross_attn_input: torch.Tensor = None) -> torch.Tensor:
        # ... (forward 方法其余部分保持不变) ...
        # Self-attention block (with pre-layer norm)
        residual = x
        x_norm = self.norm1(x)
        self_attn_output, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = residual + self.dropout_layer(self_attn_output)
        
        # Cross-attention block if external context is provided
        if cross_attn_input is not None:
            residual = x
            x_norm = self.norm2(x)
            # key 和 value 都使用 cross_attn_input
            cross_attn_output, _ = self.cross_attn(x_norm, cross_attn_input, cross_attn_input)
            x = residual + self.dropout_layer(cross_attn_output)
        
        # Feed-forward block (with pre-layer norm)
        residual = x
        x_norm = self.norm3(x)
        ff_output = self.ff(x_norm)
        x = residual + self.dropout_layer(ff_output)
        
        return x


class HybridAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int = 2048,
        seq_len: int = 77,
        mlp_hidden_dim: int = 4096,
        num_transformer_layers: int = 3,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        cross_attn_dim: int = 768
    ):
        super().__init__()
        self.seq_len = seq_len
        self.cross_attn_dim = cross_attn_dim

        # Stage 1: MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(mlp_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, input_dim),
            nn.LayerNorm(input_dim)
        )

        # Stage 2: Positional encoding
        self.position_embed = nn.Parameter(
            torch.randn(1, seq_len, input_dim) * 0.02
        )

        # Stage 3: Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlockWithCrossAttn(
                input_dim=input_dim,
                num_heads=num_attention_heads,
                hidden_dim=mlp_hidden_dim,
                dropout=dropout,
                cross_attn_dim=cross_attn_dim
            )
            for _ in range(num_transformer_layers)
        ])

        self._init_weights()  # 在所有参数定义后调用

    def _init_weights(self):
        # MLP initialization
        nn.init.eye_(self.mlp[4].weight)
        nn.init.zeros_(self.mlp[4].bias)

        # Initialize Transformer blocks' weights
        for block in self.transformer_blocks:
            # Self-attention weights
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            nn.init.constant_(block.self_attn.in_proj_bias, 0.)
            # Cross-attention weights
            # 现在手动初始化我们创建的权重
            nn.init.xavier_uniform_(block.q_proj_weight)
            nn.init.xavier_uniform_(block.k_proj_weight)
            nn.init.xavier_uniform_(block.v_proj_weight)

            nn.init.constant_(block.cross_attn.in_proj_bias, 0.) # bias 保持不变
            # Feed-forward layers
            for m in block.ff:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def extend_positional_encoding(self, new_seq_len: int) -> torch.Tensor:
        # ... (extend_positional_encoding 方法保持不变) ...
        """
        将已有位置编码扩展到新的长度 new_seq_len。
        这里采用线性插值方式扩展 [1, old_seq_len, input_dim] -> [1, new_seq_len, input_dim]
        """
        old_pos_embed = self.position_embed  # [1, old_seq_len, input_dim]
        old_seq_len = old_pos_embed.shape[1]
        if old_seq_len == new_seq_len:
            return old_pos_embed
        # 转置至 [1, input_dim, old_seq_len]以便沿序列维度插值
        new_pos_embed = F.interpolate(
            old_pos_embed.transpose(1, 2),  # [1, input_dim, old_seq_len]
            size=new_seq_len,
            mode='linear',  # 或 'nearest'，根据需要选择
            align_corners=False
        ).transpose(1, 2)
        return new_pos_embed
    def forward(self, x: torch.Tensor, cross_attn_input: torch.Tensor = None) -> torch.Tensor:
        # ... (forward 方法保持不变)
        """
        输入:
            x: [batch_size, num_tokens, input_dim] (当LLM输出多个token时)
            或 [batch_size, input_dim] (单token输出)
            cross_attn_input: [batch_size, seq_length_context, input_dim] (可选外部上下文信息)
        输出:
            [batch_size, seq_len, input_dim] (适配后的条件序列)
        """
        batch_size = x.shape[0]
        
        # 如果输入只有单个token，则扩展为序列
        if x.ndim == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_dim]
            x = x.repeat(1, self.seq_len, 1)  # [batch, seq_len, input_dim]
        else:
            # 如果输入 token 数与期望的不一致，可以通过插值或其他方式调整
            # 这里假设输入token数和 self.seq_len 一致，否则需要额外处理
            pass
        
        # 添加（或更新）位置编码
        x = x + self.position_embed
        
        # 通过MLP进行初步非线性变换
        x = self.mlp(x)
        
        # 通过Transformer Blocks，传入可选的 cross attention 信息
        for block in self.transformer_blocks:
            x = block(x, cross_attn_input=cross_attn_input)
        
        return x  # [batch, seq_len, input_dim]