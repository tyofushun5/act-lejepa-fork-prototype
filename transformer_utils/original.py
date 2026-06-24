import torch
from torch import nn

from transformers.activations import ACT2FN

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.dropout1 = nn.Dropout(config.attention_dropout)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout2 = nn.Dropout(config.attention_dropout)
    
    def forward(self, x):
        x = self.up_proj(x)
        x = self.act_fn(x)
        x = self.dropout1(x)
        x = self.down_proj(x)
        x = self.dropout2(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_interface = nn.MultiheadAttention(
            config.hidden_size, 
            config.num_attention_heads, 
            config.attention_dropout, 
            batch_first=True
        )

    def forward(self, x: torch.Tensor, attention_mask=None):
        # Expected attention mask can have shape `(B, nh, T, T)`.
        # True indicates that the token is visible, False that it isn't.
        # However, nn.MultiheadAttention expects a boolean mask where True indicates masked, and 
        # False indicates visible. Thus, we have to invert the given attention mask.
        # Additionally, it requires the input attention mask to have shape `(B * nh, T, T)`.
        if attention_mask is not None:
            assert attention_mask.ndim == 4 and attention_mask.dtype == torch.bool
            attention_mask = ~attention_mask.flatten(0, 1) # (B * nh, T, T)

        q = k = v = x
        attention_output, _ = self.attention_interface(
            q, k, v, attn_mask=attention_mask
        )
        return attention_output # (B T C)


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_interface = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_dropout,
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor, memory: torch.Tensor, attention_mask=None):
        # Expected attention mask can have shape `(B, nh, T, T)`.
        # True indicates that the token is visible, False that it isn't.
        # However, nn.MultiheadAttention expects a boolean mask where True indicates masked, and 
        # False indicates visible. Thus, we have to invert the given attention mask.
        # Additionally, it requires the input attention mask to have shape `(B * nh, T, T)`.
        if attention_mask is not None:
            assert attention_mask.ndim == 4 and attention_mask.dtype == torch.bool
            attention_mask = ~attention_mask.flatten(0, 1) # (B * nh, T, T)
        
        q = x
        k = v = memory
        attention_output, _ = self.attention_interface(
            q, k, v, attn_mask=attention_mask
        )
        return attention_output # (B T C)


class EncoderLayer(nn.Module):
    # alias: Block
    def __init__(self, config):
        '''Encoder layer with self-attention and MLP.'''
        super().__init__()
        self.self_attn = SelfAttention(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)

        self.mlp = MLP(config)
        self.norm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor, attention_mask=None):
        # uses post-norm in forward
        x = self.norm1(x + self.self_attn(x, attention_mask))
        x = self.norm2(x + self.mlp(x))
        return x


class DecoderLayer(nn.Module):
    # alias: CrossAttentionBlock
    def __init__(self, config):
        '''Decoder layer with self-attention, cross-attention, and MLP.'''
        super().__init__()
        self.self_attn = SelfAttention(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)

        self.cross_attn = CrossAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size)

        self.mlp = MLP(config)
        self.norm3 = nn.LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, attention_mask=None, cross_attention_mask=None):
        # uses post-norm in forward
        x = self.norm1(x + self.self_attn(x, attention_mask))
        x = self.norm2(x + self.cross_attn(x, memory, cross_attention_mask))
        x = self.norm3(x + self.mlp(x))
        return x
