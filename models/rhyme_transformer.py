"""
押韵+格律监督微调 Transformer
在预训练 CharTransformer 上加两个辅助分类头:
  - 韵母头: 预测末字的韵部 (num_rhyme_classes 类)
  - 平仄头: 预测每字的平/仄 (2 类)
推理时辅助头不参与, 仅 LM head 生成。
"""
import torch
import torch.nn as nn
import config
from models.transformer import CharTransformer


class RhymeToneTransformer(nn.Module):
    def __init__(self, vocab_size, num_rhyme_classes, d_model=None, nhead=None,
                 num_layers=None, dim_ffn=None, max_seq_len=None, dropout=None):
        super().__init__()
        self.num_rhyme_classes = num_rhyme_classes

        # 基础 Transformer（复用 CharTransformer 架构）
        self.transformer = CharTransformer(
            vocab_size,
            d_model=d_model or config.T_D_MODEL,
            nhead=nhead or config.T_NHEAD,
            num_layers=num_layers or config.T_LAYERS,
            dim_ffn=dim_ffn or config.T_FFN,
            max_seq_len=max_seq_len or config.MAX_SEQ_LEN,
            dropout=dropout or config.DROPOUT,
        )
        hidden = self.transformer.d_model

        # ── 辅助分类头 ──
        # 韵母分类器: 预测末字属于哪个韵部
        self.rhyme_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden, num_rhyme_classes),
        )

        # 平仄分类器: 预测每个位置是平(0)还是仄(1)
        self.tone_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden // 2, 2),
        )

        self._init_aux_weights()

    def _init_aux_weights(self):
        for module in [self.rhyme_head, self.tone_head]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x, past_kvs=None, use_cache=False):
        """
        训练 (use_cache=False): 返回 (lm_logits, rhyme_logits, tone_logits)
        推理 (use_cache=True):  返回 (lm_logits, new_kvs)  ← 与 CharTransformer 兼容
        """
        # 从基础 Transformer 获取 hidden states
        result = self.transformer(x, past_kvs=past_kvs,
                                   use_cache=use_cache, return_hidden=True)
        if use_cache:
            (lm_logits, new_kvs), hidden = result
            return lm_logits, new_kvs           # 推理只走 LM head
        else:
            lm_logits, hidden = result

        # 辅助头只在训练时使用
        rhyme_logits = self.rhyme_head(hidden)     # (B, T, num_rhyme_classes)
        tone_logits = self.tone_head(hidden)       # (B, T, 2)
        return lm_logits, rhyme_logits, tone_logits

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        base = self.transformer.count_parameters()
        aux = total - base
        return total  # 供外部使用, aux 单独标注

    def count_aux_parameters(self):
        """仅辅助头参数量"""
        aux_params = list(self.rhyme_head.parameters()) + list(self.tone_head.parameters())
        return sum(p.numel() for p in aux_params if p.requires_grad)
