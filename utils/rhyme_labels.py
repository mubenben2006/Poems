"""
韵母 / 平仄标签提取: 为韵律监督微调提供逐位置训练信号
"""
import os, json
from collections import defaultdict
from pypinyin import pinyin, Style
import torch
import torch.nn.functional as F
import config

# 四种标准七绝平仄格式 (平=0, 仄=1)
_TONE_PATTERNS = [
    [[0,0,1,1,1,0,0], [1,1,0,0,1,1,0], [1,1,0,0,0,1,1], [0,0,1,1,1,0,0]],
    [[1,1,0,0,0,1,1], [0,0,1,1,0,0,1], [0,0,1,1,1,0,0], [1,1,0,0,0,1,1]],
    [[0,0,1,1,0,0,1], [1,1,0,0,0,1,1], [1,1,0,0,1,1,0], [0,0,1,1,0,0,1]],
    [[1,1,0,0,1,1,0], [0,0,1,1,0,0,1], [0,0,1,1,1,0,0], [1,1,0,0,0,1,1]],
]


# ═══════════════════════════════════════════════════════════════
# 韵母分类
# ═══════════════════════════════════════════════════════════════

def build_rhyme_classes():
    """扫描词表中所有汉字，按 pypinyin 韵母分组。返回 (char2class, num_classes, class_names)"""
    tok_path = os.path.join(config.DATA_DIR, 'tokenizer.json')
    with open(tok_path, 'r', encoding='utf-8') as f:
        char2idx = json.load(f)['char2idx']

    final2chars = defaultdict(set)
    for ch in char2idx:
        if ch in config.SPECIAL_TOKENS:
            continue
        py = pinyin(ch, style=Style.FINALS, strict=False)
        if py and py[0] and py[0][0]:
            final2chars[py[0][0]].add(ch)
        else:
            final2chars['__unknown__'].add(ch)

    final2chars = {f: cs for f, cs in final2chars.items() if len(cs) >= 5}
    sorted_finals = sorted(final2chars.items(), key=lambda x: -len(x[1]))

    char2class = {}
    class_names = []
    for cls_id, (final, chars) in enumerate(sorted_finals):
        class_names.append(final)
        for ch in chars:
            char2class[ch] = cls_id

    print(f"韵母分类: {len(class_names)} 类 (覆盖 {len(char2class)} 字)")
    return char2class, len(class_names), class_names


# ═══════════════════════════════════════════════════════════════
# 平仄
# ═══════════════════════════════════════════════════════════════

def get_tone_label(ch):
    """返回 0=平, 1=仄, -1=未知"""
    py = pinyin(ch, style=Style.TONE3, strict=False)
    if py and py[0] and py[0][0]:
        s = py[0][0]
        if s[-1].isdigit():
            t = int(s[-1])
            return 0 if t in (1, 2) else 1
    return -1


# ═══════════════════════════════════════════════════════════════
# 批量标签生成（用于 aux head 训练，当前未使用）
# ═══════════════════════════════════════════════════════════════

def extract_batch_labels(poem_lines_list, tokenizer, char2rhyme_class, seq_len=None):
    seq_len = seq_len or config.MAX_SEQ_LEN
    B = len(poem_lines_list)
    eol_positions = [7, 15, 23, 31]

    rhyme_labels = torch.full((B, seq_len), -1, dtype=torch.long)
    tone_labels = torch.full((B, seq_len), -1, dtype=torch.long)
    rhyme_mask = torch.zeros(B, seq_len)
    tone_mask = torch.zeros(B, seq_len)

    for b, poem_lines in enumerate(poem_lines_list):
        tokens = [config.BOS_IDX]
        for i, line in enumerate(poem_lines):
            for ch in line:
                tokens.append(tokenizer.char2idx.get(ch, config.UNK_IDX))
            tokens.append(config.SEP_IDX if i < 3 else config.EOS_IDX)

        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]

        for t in range(min(len(tokens) - 1, seq_len)):
            next_token = tokens[t + 1]
            next_char = tokenizer.idx2char.get(next_token, '')
            if t + 1 in eol_positions:
                rc = char2rhyme_class.get(next_char, -1)
                if rc >= 0:
                    rhyme_labels[b, t] = rc
                    rhyme_mask[b, t] = 1.0
            if next_char not in config.SPECIAL_TOKENS and next_char != config.UNK_TOKEN:
                tl = get_tone_label(next_char)
                if tl >= 0:
                    tone_labels[b, t] = tl
                    tone_mask[b, t] = 1.0

    return rhyme_labels, tone_labels, rhyme_mask, tone_mask


# ═══════════════════════════════════════════════════════════════
# 对比 Loss 掩码矩阵（直接作用在 LM logits 上）
# ═══════════════════════════════════════════════════════════════

def build_contrastive_masks(tokenizer, char2rhyme_class, num_rhyme_classes):
    """构建词表级掩码 + token→韵类/声调查询表"""
    V = tokenizer.vocab_size
    rhyme_mat = torch.zeros(num_rhyme_classes, V)
    tone_mat = torch.zeros(2, V)
    token2rhyme = [-1] * V
    token2tone = [-1] * V

    for idx in range(V):
        ch = tokenizer.idx2char.get(idx, '')
        if ch in config.SPECIAL_TOKENS or ch == config.UNK_TOKEN:
            continue
        rc = char2rhyme_class.get(ch, -1)
        if rc >= 0:
            rhyme_mat[rc, idx] = 1.0
            token2rhyme[idx] = rc
        tl = get_tone_label(ch)
        if tl >= 0:
            tone_mat[tl, idx] = 1.0
            token2tone[idx] = tl

    return rhyme_mat, tone_mat, token2rhyme, token2tone


def contrastive_rhyme_loss(lm_logits, y, mask, rhyme_mask_mat, token2rhyme):
    """韵母对比 loss: 末字位置 (6,14,22,30)，-log(Σ prob[同韵字])。全向量化。"""
    B, T, V = lm_logits.shape
    device = lm_logits.device
    probs = F.softmax(lm_logits, dim=-1)
    rhyme_mat = rhyme_mask_mat.to(device)                    # (C, V)

    # same_rhyme[b,t,c] = 韵类 c 在位置 (b,t) 的概率质量
    same_rhyme = probs @ rhyme_mat.T                         # (B, T, C)

    eol_positions = [6, 14, 22, 30]
    eol_mask = torch.zeros(B, T, device=device)
    for pos in eol_positions:
        if pos < T:
            eol_mask[:, pos] = 1.0
    eol_mask = eol_mask * mask

    if eol_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    # token2rhyme 转为 tensor，查表: y → rhyme class
    t2r = torch.tensor(token2rhyme, device=device)           # (V,)
    rc = t2r[y.clamp(0, V - 1)]                              # (B, T)
    p_same = same_rhyme.gather(-1, rc.clamp(min=0).unsqueeze(-1)).squeeze(-1)

    loss = -torch.log(p_same.clamp(min=1e-8))
    return (loss * eol_mask).sum() / eol_mask.sum().clamp(min=1)


def contrastive_tone_loss(lm_logits, y, mask, tone_mask_mat, token2tone):
    """平仄对比 loss: 每个汉字位置，-log(Σ prob[正确声调字])。全向量化。"""
    B, T, V = lm_logits.shape
    device = lm_logits.device
    probs = F.softmax(lm_logits, dim=-1)
    tone_mat = tone_mask_mat.to(device)                      # (2, V)

    # correct_tone[b,t,k] = 声调 k 在位置 (b,t) 的概率质量
    correct_tone = probs @ tone_mat.T                        # (B, T, 2)

    if mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    t2t = torch.tensor(token2tone, device=device)            # (V,)
    tl = t2t[y.clamp(0, V - 1)]                              # (B, T)
    p_correct = correct_tone.gather(-1, tl.clamp(min=0).unsqueeze(-1)).squeeze(-1)

    loss = -torch.log(p_correct.clamp(min=1e-8))
    return (loss * mask).sum() / mask.sum().clamp(min=1)
