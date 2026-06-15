"""
GPT 风格 Decoder-only Transformer（带 KV-Cache）
架构: TokenEmb + PosEmb → N×DecoderBlock(Pre-Norm) → LayerNorm → Linear
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class CausalSelfAttention(nn.Module):
    """带因果 mask 的多头自注意力, 支持 KV-Cache"""

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)              # (3, B, nh, T, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if past_kv is not None:                        # 拼接历史 KV
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        new_kv = (k, v) if use_cache else None

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale

        if past_kv is None:                            # 仅全序列时需要 causal mask
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out), new_kv


class DecoderBlock(nn.Module):
    """Pre-Norm Decoder 层: norm → attn → + → norm → ffn → +"""

    def __init__(self, d_model, nhead, dim_ffn, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nhead, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, new_kv = self.attn(self.norm1(x), past_kv, use_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_kv


class CharTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=None, nhead=None, num_layers=None,
                 dim_ffn=None, max_seq_len=None, dropout=None):
        super().__init__()
        self.d_model = d_model or config.T_D_MODEL
        self.nhead = nhead or config.T_NHEAD
        self.num_layers = num_layers or config.T_LAYERS
        self.max_seq_len = max_seq_len or config.MAX_SEQ_LEN
        self.dropout_rate = dropout or config.DROPOUT

        self.token_embedding = nn.Embedding(vocab_size, self.d_model,
                                            padding_idx=config.PAD_IDX)

        # 正弦位置编码 (可索引任意 offset)
        pe = torch.zeros(self.max_seq_len, self.d_model)
        pos = torch.arange(0, self.max_seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, self.d_model, 2).float() *
                        (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

        self.dropout = nn.Dropout(self.dropout_rate)

        self.layers = nn.ModuleList([
            DecoderBlock(self.d_model, self.nhead,
                        dim_ffn or config.T_FFN, self.dropout_rate)
            for _ in range(self.num_layers)
        ])
        self.final_norm = nn.LayerNorm(self.d_model)
        self.output = nn.Linear(self.d_model, vocab_size)

        # Weight tying: 输出投影 = 输入嵌入（减少参数 + 正则化）
        self.output.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        nn.init.zeros_(self.output.bias)

    def tie_weights(self):
        """将输出投影权重重新绑定到嵌入（checkpoint 加载后调用）"""
        self.output.weight = self.token_embedding.weight

    def forward(self, x, past_kvs=None, use_cache=False, return_hidden=False):
        """
        x: (B, T) token ids
        past_kvs: 可选, KV-Cache 列表
        return_hidden: 返回 final_norm 后的 hidden states（用于辅助头）
        训练时返回 logits; 推理时返回 (logits, new_kvs)
        """
        B, T = x.shape
        x = self.token_embedding(x) * math.sqrt(self.d_model)

        # 位置编码: 支持 KV-Cache 的 offset
        offset = past_kvs[0][0].size(2) if past_kvs else 0
        x = x + self.pe[:, offset:offset + T, :]
        x = self.dropout(x)

        new_kvs = []
        for i, layer in enumerate(self.layers):
            pkv = past_kvs[i] if past_kvs else None
            x, nkv = layer(x, past_kv=pkv, use_cache=use_cache)
            new_kvs.append(nkv)

        hidden = self.final_norm(x)
        logits = self.output(hidden)

        if return_hidden:
            if use_cache:
                return (logits, new_kvs), hidden
            return logits, hidden
        return (logits, new_kvs) if use_cache else logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
