"""
数据下载、筛选、加载 —— 只取七言绝句（4句×7字）
数据来源: DICAlab Ancient Poems Dataset (232,670首绝句)
"""
import os
import json
import re
import random
import torch
from torch.utils.data import Dataset, DataLoader
import config


# ============================================================
# 下载 & 解析
# ============================================================

def download_dataset():
    """从 Google Drive 下载数据集，已存在则跳过"""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    save_path = os.path.join(config.DATA_DIR, "poet_tang.json")
    if os.path.exists(save_path):
        print(f"数据集已存在: {save_path}")
        return save_path

    import gdown
    url = f"https://drive.google.com/uc?id={config.GDRIVE_FILE_ID}"
    print("正在下载数据集...")
    gdown.download(url, save_path, quiet=False)
    return save_path


def _parse_poem(line_text):
    """用中文标点分割一行诗，只保留汉字"""
    parts = re.split(r'[，。？！；、：,]', line_text)
    result = []
    for p in parts:
        chars = re.findall(r'[一-鿿]', p)
        if chars:
            result.append("".join(chars))
    return result


def load_and_filter(file_path):
    """加载并筛选：只保留4句×7字的七言绝句"""
    with open(file_path, "r", encoding="utf-8") as f:
        raw = [l.strip() for l in f if l.strip()]

    poems, skipped = [], 0
    for line in raw:
        parts = _parse_poem(line)
        if len(parts) == 4 and all(len(p) == 7 for p in parts):
            poems.append(parts)
        else:
            skipped += 1

    print(f"七言绝句: {len(poems):,} 首, 跳过: {skipped:,} 首")
    return poems


def prepare_data():
    """完整数据准备: 下载 → 筛选 → 保存 → 返回"""
    raw_path = download_dataset()
    poems = load_and_filter(raw_path)

    processed = os.path.join(config.DATA_DIR, "qiyan_jueju.json")
    with open(processed, "w", encoding="utf-8") as f:
        json.dump(poems, f, ensure_ascii=False, indent=2)
    print(f"已保存: {processed}")
    return poems


# ============================================================
# 序列化
# ============================================================

def poem_to_sequence(poem_lines):
    """将四句诗转为: <BOS>句1<SEP>句2<SEP>句3<SEP>句4<EOS>"""
    parts = [config.BOS_TOKEN]
    for i, line in enumerate(poem_lines):
        parts.append(line)
        parts.append(config.SEP_TOKEN if i < 3 else config.EOS_TOKEN)
    return "".join(parts)


# ============================================================
# PyTorch Dataset
# ============================================================

class PoetryDataset(Dataset):
    """字符级 LM 数据集: x=tokens[:-1], y=tokens[1:]"""

    def __init__(self, poems, tokenizer, seq_len=config.MAX_SEQ_LEN):
        self.seq_len = seq_len
        self.data = []
        for poem in poems:
            tokens = tokenizer.encode(poem_to_sequence(poem))
            # 截断或填充
            if len(tokens) > seq_len:
                tokens = tokens[:seq_len]
            else:
                tokens += [config.PAD_IDX] * (seq_len - len(tokens))
            self.data.append(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        t = self.data[idx]
        x = torch.tensor(t[:-1], dtype=torch.long)
        y = torch.tensor(t[1:], dtype=torch.long)
        mask = (y != config.PAD_IDX).float()
        return x, y, mask


def collate_fn(batch):
    """批次内动态 padding"""
    x_list, y_list, m_list = zip(*batch)
    max_len = max(x.size(0) for x in x_list)

    def pad(t_list, val):
        return torch.stack([
            torch.cat([t, t.new_full((max_len - t.size(0),), val)])
            if t.size(0) < max_len else t
            for t in t_list
        ])

    return pad(x_list, config.PAD_IDX), pad(y_list, config.PAD_IDX), pad(m_list, 0)


def create_dataloaders(poems, tokenizer, batch_size=None):
    """划分训练/验证集，返回 DataLoader"""
    batch_size = batch_size or config.BATCH_SIZE

    n = len(poems)
    n_train = int(n * config.TRAIN_SPLIT)
    random.seed(42)
    shuffled = poems.copy()
    random.shuffle(shuffled)

    train_poems = shuffled[:n_train]
    val_poems = shuffled[n_train:]

    if config.MAX_TRAIN and len(train_poems) > config.MAX_TRAIN:
        train_poems = train_poems[:config.MAX_TRAIN]

    train_ds = PoetryDataset(train_poems, tokenizer)
    val_ds = PoetryDataset(val_poems, tokenizer)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False,
                            collate_fn=collate_fn, drop_last=False)

    print(f"训练集: {len(train_ds):,}, 验证集: {len(val_ds):,}")
    return train_loader, val_loader
