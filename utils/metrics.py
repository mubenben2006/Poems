"""
评测指标: PPL、格式合规率、自重复度、押韵率、平仄匹配率、语义相关度
"""
import os, json, math
import torch
import torch.nn.functional as F
from collections import Counter
from pypinyin import pinyin, Style
import config


# ============================================================
# PPL
# ============================================================

def compute_ppl(model, data_loader, device=None):
    """计算验证集困惑度"""
    if device is None:
        device = config.DEVICE
    model.eval()
    total_loss, total_tokens = 0.0, 0

    with torch.no_grad():
        for x, y, mask in data_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), reduction='none'
            ).view(y.size())
            loss = (loss * mask).sum() / mask.sum()
            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()

    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')


# ============================================================
# 格式合规率
# ============================================================

def compute_format_rate(generated):
    """
    generated: list[dict], 每个 dict 含 "lines" (list[str])
    返回格式合规率 (4句 × 每句7字)
    """
    if not generated:
        return 0.0
    ok = 0
    for p in generated:
        lines = p.get("lines", [])
        if len(lines) == 4 and all(len(l) == 7 for l in lines):
            ok += 1
    return ok / len(generated)


# ============================================================
# 自重复度
# ============================================================

def compute_self_repeat(lines):
    """计算 n-gram 重复率和行间最大重叠"""
    if not lines:
        return {"bigram": 0.0, "trigram": 0.0, "max_overlap": 0.0}

    all_text = "".join(lines)

    def ngrams(text, n):
        return [text[i:i+n] for i in range(len(text) - n + 1)] if len(text) >= n else []

    bg = ngrams(all_text, 2)
    tg = ngrams(all_text, 3)
    b_rep = 1 - len(set(bg)) / len(bg) if bg else 0.0
    t_rep = 1 - len(set(tg)) / len(tg) if tg else 0.0

    def lcs(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        mx = 0
        for i in range(m):
            for j in range(n):
                if a[i] == b[j]:
                    dp[i+1][j+1] = dp[i][j] + 1
                    mx = max(mx, dp[i+1][j+1])
        return mx

    max_overlap = 0.0
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            l = lcs(lines[i], lines[j])
            max_overlap = max(max_overlap, l / max(len(lines[i]), len(lines[j])))
    return {"bigram": b_rep, "trigram": t_rep, "max_overlap": max_overlap}


# ============================================================
# 押韵 & 平仄
# ============================================================

# 四种标准七绝平仄格式 (平=1, 仄=0)
_PATTERNS = [
    [[1,1,0,0,0,1,1], [0,0,1,1,0,0,1], [0,0,1,1,1,0,0], [1,1,0,0,0,1,1]],  # 平起入韵
    [[0,0,1,1,0,0,1], [1,1,0,0,0,1,1], [1,1,0,0,1,1,0], [0,0,1,1,0,0,1]],  # 仄起入韵
    [[1,1,0,0,1,1,0], [0,0,1,1,0,0,1], [0,0,1,1,1,0,0], [1,1,0,0,0,1,1]],  # 平起不入韵
    [[0,0,1,1,1,0,0], [1,1,0,0,0,1,1], [1,1,0,0,1,1,0], [0,0,1,1,0,0,1]],  # 仄起不入韵
]


def _get_tone(ch):
    """获取声调: 1/2=平, 3/4=仄, 0=未知"""
    py = pinyin(ch, style=Style.TONE3, strict=False)
    if py and py[0] and py[0][0]:
        s = py[0][0]
        if s[-1].isdigit():
            return int(s[-1])
    return 0


def _get_final(ch):
    """获取韵母（无声调）"""
    py = pinyin(ch, style=Style.FINALS, strict=False)
    return py[0][0] if py and py[0] else ''


def _get_rhyme_key(ch):
    """
    获取韵部 key：去除韵头(i/u/ü)后的韵腹+韵尾。
    用于判断传统诗词押韵（'ian'→'an', 'uan'→'an', 'iang'→'ang' 等）。
    """
    f = _get_final(ch)
    if not f:
        return ''
    # 去除韵头 (i-, u-, ü-)，保留韵腹+韵尾
    if len(f) >= 2:
        if f[0] == 'i' and f[1] not in ('u', 'o'):  # i 开头但不是 iu/io
            return f[1:]
        if f[0] == 'u' and f[1] not in ('i', 'n'):  # u 开头但不是 ui/un
            return f[1:]
        if f[0] == 'v':  # ü
            return f[1:]
    return f


def compute_rhyme_rate(generated):
    """
    押韵率：第2、4句末字必须同韵部且为平声（第3句末字仄声更好但不强制）。
    使用 _get_rhyme_key 去除韵头以匹配传统诗词押韵习惯。
    """
    if not generated:
        return 0.0
    ok = 0
    for p in generated:
        lines = p.get("lines", [])
        if len(lines) != 4 or any(len(l) != 7 for l in lines):
            continue
        l2rk, l4rk = _get_rhyme_key(lines[1][-1]), _get_rhyme_key(lines[3][-1])
        l2t, l4t = _get_tone(lines[1][-1]), _get_tone(lines[3][-1])
        if l2rk and l4rk and l2rk == l4rk and l2t in (1, 2) and l4t in (1, 2):
            ok += 1
    return ok / len(generated)


def compute_tone_rate(generated):
    """
    平仄匹配率：逐位置与四种标准格式比对，取最佳匹配。
    """
    if not generated:
        return 0.0
    total, matched = 0, 0
    for p in generated:
        lines = p.get("lines", [])
        if len(lines) != 4 or any(len(l) != 7 for l in lines):
            continue
        tones = [[1 if _get_tone(ch) in (1, 2) else 0 for ch in line] for line in lines]
        best = 0
        for pat in _PATTERNS:
            hits = sum(tones[i][j] == pat[i][j] for i in range(4) for j in range(7))
            best = max(best, hits)
        matched += best
        total += 28
    return matched / total if total > 0 else 0.0


def poem_score(lines, relevance=None):
    """综合质量评分: 押韵+平仄+相关度+多样性-重复 (越高越好)"""
    if len(lines) != 4 or any(len(l) != 7 for l in lines):
        return 0.0

    # 押韵 (权重最高)
    r2, r4 = _get_rhyme_key(lines[1][-1]), _get_rhyme_key(lines[3][-1])
    t2, t4 = _get_tone(lines[1][-1]), _get_tone(lines[3][-1])
    rhyme = 3.0 if (r2 and r4 and r2 == r4 and t2 in (1, 2) and t4 in (1, 2)) else 0.0

    # 平仄 (与四种标准格式比, 取最佳)
    tones = [[1 if _get_tone(c) in (1, 2) else 0 for c in l] for l in lines]
    best_tone = max(sum(tones[i][j] == p[i][j] for i in range(4) for j in range(7))
                    for p in _PATTERNS) / 28

    # 自重复惩罚
    rep = compute_self_repeat(lines)
    rep_pen = rep['bigram'] * 3 + rep['trigram'] * 5 + rep['max_overlap'] * 3

    # 字符多样性
    chars = ''.join(lines)
    diversity = len(set(chars)) / len(chars)

    # 语义相关度
    rel = relevance if relevance else 0.5

    return rhyme + best_tone * 2 + rel * 2 + diversity * 1.5 - rep_pen


def find_top_poems(all_results, top_n=5):
    """从所有生成结果中找出质量最高的 top_n 首"""
    scored = []
    for model_name, results in all_results:
        for r in results:
            s = poem_score(r['lines'], r.get('relevance'))
            if s > 0:
                scored.append((s, model_name, r))
    scored.sort(key=lambda x: -x[0])
    return scored[:top_n]


def debug_rhyme(generated, title="韵脚诊断"):
    """
    逐首打印末字韵母/声调/韵部详情，用于诊断押韵率。
    """
    print(f"\n  {'='*60}")
    print(f"  【{title}】")
    for i, p in enumerate(generated):
        lines = p.get("lines", [])
        if len(lines) != 4 or any(len(l) != 7 for l in lines):
            print(f"  [{i}] 格式不完整, 跳过")
            continue
        l1c, l2c = lines[0][-1], lines[1][-1]
        l3c, l4c = lines[2][-1], lines[3][-1]
        l1rk, l2rk = _get_rhyme_key(l1c), _get_rhyme_key(l2c)
        l3rk, l4rk = _get_rhyme_key(l3c), _get_rhyme_key(l4c)
        l1t, l2t = _get_tone(l1c), _get_tone(l2c)
        l3t, l4t = _get_tone(l3c), _get_tone(l4c)
        tl = lambda t: '平' if t in (1,2) else ('仄' if t in (3,4) else '?')
        rhyme_ok = (l2rk == l4rk and l2t in (1,2) and l4t in (1,2))
        print(f"  [{i}] 末字: {l1c}(韵={l1rk}/{tl(l1t)}) "
              f"{l2c}(韵={l2rk}/{tl(l2t)}) "
              f"{l3c}(韵={l3rk}/{tl(l3t)}) "
              f"{l4c}(韵={l4rk}/{tl(l4t)}) "
              f"→ {'✓' if rhyme_ok else '✗ 不押韵'}")
        print(f"       {lines[0]} | {lines[1]} | {lines[2]} | {lines[3]}")
    print()

# ============================================================
# 语义相关度（IDF加权嵌入余弦相似度）
# ============================================================

# 只加载一次的缓存
_idf_cache = None
_emb_cache = {}           # model_name → embeddings, 不同模型独立缓存


def _load_idf():
    """从训练语料构建字符级 IDF（只算一次）"""
    global _idf_cache
    if _idf_cache is not None:
        return _idf_cache

    corpus_path = os.path.join(config.DATA_DIR, 'qiyan_jueju.json')
    with open(corpus_path, 'r', encoding='utf-8') as f:
        poems = json.load(f)

    # 每句作为"文档"
    lines = []
    for poem in poems:
        for line in poem:
            lines.append(line)

    N = len(lines)
    df = Counter()
    for line in lines:
        for ch in set(line):
            df[ch] += 1

    _idf_cache = {ch: math.log(N / (1.0 + count)) for ch, count in df.items()}
    print(f"IDF 已构建: {len(_idf_cache)} 字, "
          f"范围 [{min(_idf_cache.values()):.1f}, {max(_idf_cache.values()):.1f}]")
    return _idf_cache


def _load_embeddings(model_name):
    """从 checkpoint 提取字嵌入矩阵（按模型名缓存）"""
    global _emb_cache
    if model_name in _emb_cache:
        return _emb_cache[model_name]

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f'{model_name}_best.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f'{model_name}_final.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 {model_name} checkpoint 用于提取 embedding")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict']
    # 按优先级找嵌入权重: LSTM/Transformer → rhyme_transformer 嵌套
    for key in ['embedding.weight', 'token_embedding.weight',
                'transformer.token_embedding.weight']:
        if key in state:
            _emb_cache[model_name] = state[key].numpy()
            print(f"已加载 {model_name} 字嵌入: {_emb_cache[model_name].shape}")
            return _emb_cache[model_name]
    raise KeyError("未在 checkpoint 中找到 embedding 权重")


def compute_relevance(first_line, generated_lines, model_name='lstm'):
    """
    语义相关度：IDF 加权字嵌入余弦相似度。

    原理：
      1. 从训练好的 checkpoint 提取 Embedding 权重（每个汉字→128维向量）
      2. 用训练语料 IDF 加权：罕见实义词（如"翠" IDF≈7）贡献更大，
         高频虚字（如"不" IDF≈2.6）被抑制
      3. L2 归一化后计算句向量余弦相似度
      4. 返回首句与各生成句相似度的均值 ∈ [0, 1]

    Args:
        first_line: str, 首句（7字）
        generated_lines: list[str], 生成的后三句
        model_name: 'lstm' 或 'transformer'

    Returns:
        float, 语义相关度 (0~1)
    """
    if not generated_lines:
        return 0.0

    embeddings = _load_embeddings(model_name)
    idf = _load_idf()

    # 加载词表
    tok_path = os.path.join(config.DATA_DIR, 'tokenizer.json')
    with open(tok_path, 'r', encoding='utf-8') as f:
        char2idx = json.load(f)['char2idx']

    embed_dim = embeddings.shape[1]

    def sent_vec(text):
        """IDF 加权平均句向量，L2 归一化"""
        vec = [0.0] * embed_dim
        total_w = 0.0
        for ch in text:
            idx = char2idx.get(ch)
            if idx is not None and idx < len(embeddings):
                w = idf.get(ch, 1.0)
                emb = embeddings[idx]
                for d in range(embed_dim):
                    vec[d] += emb[d] * w
                total_w += w
        if total_w == 0:
            return None
        # 除以总权重
        vec = [v / total_w for v in vec]
        # L2 归一化 → dot product = cosine similarity
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm > 0 else None

    v1 = sent_vec(first_line)
    if v1 is None:
        return 0.0

    scores = []
    for line in generated_lines:
        v2 = sent_vec(line)
        if v2 is None:
            scores.append(0.0)
        else:
            dot = sum(a * b for a, b in zip(v1, v2))
            scores.append((1.0 + dot) / 2.0)  # [-1,1] → [0,1]

    return sum(scores) / len(scores)
