"""
训练脚本: 训练 LSTM 或 Transformer 七言绝句生成模型
直接运行: python train.py           （使用 config.py 中的默认值）
指定参数: python train.py --model transformer --epochs 20
"""
import os, json, math, time, sys
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt

import config
from utils.tokenizer import Tokenizer
from utils.data_loader import prepare_data, create_dataloaders, PoetryDataset
from utils.rhyme_labels import build_rhyme_classes, extract_batch_labels
from models.lstm import CharLSTM
from models.transformer import CharTransformer
from models.rhyme_transformer import RhymeToneTransformer


def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, device, is_lstm):
    model.train()
    total_loss, total_tokens = 0.0, 0
    hidden = None
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        bsz = x.size(0)

        if is_lstm:
            if hidden is None or hidden[0].size(1) != bsz:
                hidden = model.init_hidden(bsz, device)
            else:
                hidden = (hidden[0].detach(), hidden[1].detach())
            logits, hidden = model(x, hidden)
        else:
            logits = model(x)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction='none'
        ).view(y.size())
        loss = (loss * mask).sum() / mask.sum()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * mask.sum().item()
        total_tokens += mask.sum().item()
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')


@torch.no_grad()
def validate(model, loader, device, is_lstm):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    hidden = None
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        bsz = x.size(0)
        if is_lstm:
            if hidden is None or hidden[0].size(1) != bsz:
                hidden = model.init_hidden(bsz, device)
            logits, hidden = model(x, hidden)
        else:
            logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction='none'
        ).view(y.size())
        loss = (loss * mask).sum() / mask.sum()
        total_loss += loss.item() * mask.sum().item()
        total_tokens += mask.sum().item()
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')


def plot_curves(history, save_path):
    epochs = [h['epoch'] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, [h['train_ppl'] for h in history], 'b-', label='Train PPL')
    ax1.plot(epochs, [h['val_ppl'] for h in history], 'r-', label='Val PPL')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Perplexity')
    ax1.legend(); ax1.grid(True); ax1.set_title('PPL')
    ax2.plot(epochs, [math.log(h['train_ppl']) for h in history], 'b-', label='Train Loss')
    ax2.plot(epochs, [math.log(h['val_ppl']) for h in history], 'r-', label='Val Loss')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Cross Entropy')
    ax2.legend(); ax2.grid(True); ax2.set_title('Loss')
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"训练曲线: {save_path}")


def main():
    # CLI 参数全可选，默认走 config.py
    if len(sys.argv) == 1:
        model_name = config.MODEL
        epochs = config.EPOCHS
        batch_size = config.BATCH_SIZE
        lr = config.LR
        patience = config.EARLY_STOP_PATIENCE
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', default=config.MODEL, choices=['lstm','transformer'])
        parser.add_argument('--epochs', type=int, default=config.EPOCHS)
        parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
        parser.add_argument('--lr', type=float, default=config.LR)
        parser.add_argument('--patience', type=int, default=config.EARLY_STOP_PATIENCE)
        args = parser.parse_args()
        model_name = args.model; epochs = args.epochs
        batch_size = args.batch_size; lr = args.lr; patience = args.patience

    set_seed(42)
    device = config.DEVICE
    is_lstm = model_name == 'lstm'
    print(f"设备: {device} | 模型: {model_name.upper()}")

    # ---- 数据 ----
    poems = prepare_data()
    tokenizer = Tokenizer()
    tok_path = os.path.join(config.DATA_DIR, "tokenizer.json")
    if os.path.exists(tok_path):
        tokenizer.load(tok_path)
    else:
        texts = ["".join(p) for p in poems]
        tokenizer.build_from_texts(texts)
        tokenizer.save(tok_path)

    train_loader, val_loader = create_dataloaders(poems, tokenizer, batch_size)

    # ---- 模型 ----
    if is_lstm:
        model = CharLSTM(tokenizer.vocab_size).to(device)
    else:
        model = CharTransformer(tokenizer.vocab_size).to(device)
    n = model.count_parameters()
    print(f"参数量: {n:,} ({n/1e6:.2f}M)")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.LR_STEP, gamma=config.LR_GAMMA)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # ---- 断点续训 ----
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f'{model_name}_best.pt')
    start_epoch = 1
    history, best_ppl = [], float('inf')
    patience_counter = 0

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if hasattr(model, 'tie_weights'):
            model.tie_weights()
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            except Exception:
                pass
        start_epoch = ckpt.get('epoch', 0) + 1
        history = ckpt.get('history', [])
        best_ppl = ckpt.get('val_ppl', float('inf'))
        print(f"从 epoch {start_epoch} 续训 (best PPL={best_ppl:.1f})")

    t0 = time.time()

    # ---- 训练循环 ----
    for epoch in range(start_epoch, epochs + 1):
        train_ppl = train_epoch(model, train_loader, optimizer, device, is_lstm)
        val_ppl = validate(model, val_loader, device, is_lstm)
        scheduler.step()

        history.append({
            'epoch': epoch,
            'train_ppl': round(train_ppl, 2),
            'val_ppl': round(val_ppl, 2),
            'lr': optimizer.param_groups[0]['lr'],
        })

        elapsed = time.time() - t0
        eta = elapsed / (epoch - start_epoch + 1) * (epochs - epoch) if epoch > start_epoch else 0
        improved = best_ppl == float('inf') or \
                   (best_ppl - val_ppl) / best_ppl >= config.EARLY_STOP_MIN_DELTA
        flag = " *" if improved else f" ({patience_counter + 1}/{patience})" if patience > 0 else ""
        print(f"Epoch {epoch:2d}/{epochs} | Train PPL: {train_ppl:.1f} | "
              f"Val PPL: {val_ppl:.1f}{flag} | LR: {optimizer.param_groups[0]['lr']:.1e} | "
              f"{elapsed/60:.1f}m | ETA {eta/60:.1f}m")

        if improved:
            best_ppl = val_ppl
            patience_counter = 0
            ckpt = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'tokenizer_path': tok_path,
                'vocab_size': tokenizer.vocab_size,
                'epoch': epoch, 'val_ppl': val_ppl,
                'history': history,
            }
            torch.save(ckpt, os.path.join(config.CHECKPOINT_DIR, f'{model_name}_best.pt'))
        elif patience > 0:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停! 连续 {patience} 轮 Val PPL 未改善, "
                      f"最佳={best_ppl:.1f} (epoch {epoch - patience_counter})")
                break

    # ---- 保存 ----
    ckpt.update({'epoch': epochs, 'val_ppl': val_ppl,
                 'model_state_dict': model.state_dict()})
    torch.save(ckpt, os.path.join(config.CHECKPOINT_DIR, f'{model_name}_final.pt'))

    hist_path = os.path.join(config.CHECKPOINT_DIR, f'{model_name}_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    plot_curves(history, os.path.join(config.CHECKPOINT_DIR, f'{model_name}_curves.png'))
    print(f"训练完成! 最佳 Val PPL: {best_ppl:.1f}")


def finetune_rhyme_tone():
    """
    韵律监督微调: 加载预训练 Transformer → 加韵母/平仄辅助头 → 联合微调
    保存为 rhyme_transformer_best.pt
    """
    device = config.DEVICE
    print(f"\n{'='*60}")
    print("  韵律监督微调: RhymeToneTransformer")
    print('='*60)

    # ── 数据 & 标签基础设施 ──
    poems = prepare_data()
    tokenizer = Tokenizer()
    tok_path = os.path.join(config.DATA_DIR, "tokenizer.json")
    tokenizer.load(tok_path)

    poems_for_ft = poems[:config.MAX_TRAIN] if config.MAX_TRAIN else poems
    n_train = int(len(poems_for_ft) * config.TRAIN_SPLIT)
    train_poems = poems_for_ft[:n_train]
    val_poems = poems_for_ft[n_train:]

    char2class, num_rhyme_classes, _ = build_rhyme_classes()

    # ── 加载预训练 Transformer ──
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, 'transformer_best.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, 'transformer_final.pt')
    if not os.path.exists(ckpt_path):
        print("⚠ 未找到 transformer checkpoint, 请先训练 Transformer")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = RhymeToneTransformer(ckpt.get('vocab_size', tokenizer.vocab_size),
                                  num_rhyme_classes).to(device)

    # 只加载 transformer 部分的权重
    model.transformer.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.transformer.tie_weights()
    print(f"加载预训练 Transformer (epoch={ckpt['epoch']}, Val PPL={ckpt['val_ppl']:.1f})")

    base_params = model.transformer.count_parameters()
    aux_params = model.count_aux_parameters()
    total_params = model.count_parameters()
    print(f"参数量: 基础={base_params:,} + 辅助头={aux_params:,} = {total_params:,} "
          f"({total_params/1e6:.2f}M)")

    # ── 优化器（整个模型统一 LR） ──
    optimizer = torch.optim.Adam(model.parameters(), lr=config.FT_LR,
                                  weight_decay=config.WEIGHT_DECAY)

    # ── 构建对比 loss 掩码矩阵 ──
    from utils.rhyme_labels import (build_contrastive_masks, contrastive_rhyme_loss,
                                     contrastive_tone_loss)
    rhyme_mat, tone_mat, token2rhyme, token2tone = build_contrastive_masks(
        tokenizer, char2class, num_rhyme_classes)
    rhyme_mat = rhyme_mat.to(device)
    tone_mat = tone_mat.to(device)

    # ── 微调循环 ──
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_combined = float('inf')
    history = []
    ft_epochs = config.FT_EPOCHS
    t0 = time.time()

    for epoch in range(1, ft_epochs + 1):
        model.train()
        total_lm, total_rhyme, total_tone, n_batches = 0.0, 0.0, 0.0, 0

        import random
        random.seed(epoch)
        shuffled = train_poems.copy()
        random.shuffle(shuffled)

        for batch_start in range(0, len(shuffled), config.BATCH_SIZE):
            batch_poems = shuffled[batch_start:batch_start + config.BATCH_SIZE]
            if len(batch_poems) < 2:
                continue

            # 构建 token batch
            seqs, masks, tone_masks = [], [], []
            for poem in batch_poems:
                text = ''.join([config.BOS_TOKEN] +
                               [f"{line}{config.SEP_TOKEN}" if i < 3
                                else f"{line}{config.EOS_TOKEN}"
                                for i, line in enumerate(poem)])
                tokens = tokenizer.encode(text)
                if len(tokens) > config.MAX_SEQ_LEN:
                    tokens = tokens[:config.MAX_SEQ_LEN]
                else:
                    tokens += [config.PAD_IDX] * (config.MAX_SEQ_LEN - len(tokens))
                seqs.append(tokens)

                # LM mask: 目标非 PAD
                m = [1.0 if i + 1 < len(tokens) and tokens[i + 1] != config.PAD_IDX
                     else 0.0 for i in range(len(tokens) - 1)]
                masks.append(m)

                # Tone mask: 仅汉字目标 (排除 special tokens)
                tm = [1.0 if i + 1 < len(tokens) and
                      tokens[i + 1] not in (config.PAD_IDX, config.BOS_IDX,
                                             config.EOS_IDX, config.SEP_IDX,
                                             config.UNK_IDX)
                      else 0.0 for i in range(len(tokens) - 1)]
                tone_masks.append(tm)

            x = torch.tensor([s[:-1] for s in seqs], dtype=torch.long, device=device)
            y = torch.tensor([s[1:] for s in seqs], dtype=torch.long, device=device)
            mask = torch.tensor(masks, device=device)
            tone_mask = torch.tensor(tone_masks, device=device)
            B, T = x.shape

            lm_logits, rhyme_logits, tone_logits = model(x)

            # 1. LM loss
            lm_loss = F.cross_entropy(
                lm_logits.reshape(-1, lm_logits.size(-1)),
                y.reshape(-1), reduction='none'
            ).view(B, T)
            lm_loss = (lm_loss * mask).sum() / mask.sum().clamp(min=1)

            # 2. 韵母对比 loss（直接作用在 LM logits 上）
            rhyme_loss = contrastive_rhyme_loss(lm_logits, y, mask,
                                                 rhyme_mat, token2rhyme)

            # 3. 平仄对比 loss（直接作用在 LM logits 上）
            tone_loss = contrastive_tone_loss(lm_logits, y, tone_mask,
                                               tone_mat, token2tone)

            loss = lm_loss + config.FT_RHYME_WEIGHT * rhyme_loss + \
                   config.FT_TONE_WEIGHT * tone_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()

            total_lm += lm_loss.item()
            total_rhyme += rhyme_loss.item()
            total_tone += tone_loss.item()
            n_batches += 1

        avg_lm = total_lm / max(n_batches, 1)
        avg_rhyme = total_rhyme / max(n_batches, 1)
        avg_tone = total_tone / max(n_batches, 1)
        combined = avg_lm + config.FT_RHYME_WEIGHT * avg_rhyme + \
                   config.FT_TONE_WEIGHT * avg_tone

        elapsed = time.time() - t0
        improved = combined < best_combined
        flag = " *" if improved else ""
        print(f"FT Epoch {epoch:2d}/{ft_epochs} | LM:{avg_lm:.4f} "
              f"Rhyme:{avg_rhyme:.4f} Tone:{avg_tone:.4f} | "
              f"Combined:{combined:.4f}{flag} | {elapsed/60:.1f}m")

        history.append({'epoch': epoch, 'lm_loss': round(avg_lm, 4),
                        'rhyme_loss': round(avg_rhyme, 4),
                        'tone_loss': round(avg_tone, 4),
                        'combined': round(combined, 4)})

        if improved:
            best_combined = combined
            torch.save({
                'model_state_dict': model.state_dict(),
                'transformer_state_dict': model.transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'tokenizer_path': tok_path,
                'vocab_size': tokenizer.vocab_size,
                'num_rhyme_classes': num_rhyme_classes,
                'epoch': epoch, 'combined_loss': combined,
                'history': history,
            }, os.path.join(config.CHECKPOINT_DIR, 'rhyme_transformer_best.pt'))

    # 保存最终
    torch.save({
        'model_state_dict': model.state_dict(),
        'transformer_state_dict': model.transformer.state_dict(),
        'tokenizer_path': tok_path,
        'vocab_size': tokenizer.vocab_size,
        'num_rhyme_classes': num_rhyme_classes,
        'epoch': ft_epochs, 'combined_loss': combined,
        'history': history,
    }, os.path.join(config.CHECKPOINT_DIR, 'rhyme_transformer_final.pt'))

    hist_path = os.path.join(config.CHECKPOINT_DIR, 'rhyme_transformer_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"韵律微调完成! 最佳 Combined Loss: {best_combined:.4f}")
    return best_combined


if __name__ == '__main__':
    main()
