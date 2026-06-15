"""
一键运行 —— python run.py
MODE: full | train | generate | eval (config.py)
"""
import os, sys, json, math, time
import torch
import config
from utils.tokenizer import Tokenizer
from utils.sampler import Sampler
from utils.data_loader import prepare_data, create_dataloaders
from utils.metrics import (compute_ppl, compute_format_rate, compute_self_repeat,
                            compute_rhyme_rate, compute_tone_rate, compute_relevance,
                            debug_rhyme)
from generate import (load_model, generate_first_line, compute_all_metrics,
                       run_generation, divider, print_poem, print_metrics_table)
import train

# ═══════════════ 训练 ═══════════════

def run_training(model_name):
    divider(f"训练 {model_name.upper()}")
    device = config.DEVICE
    is_lstm = model_name == 'lstm'
    poems = prepare_data()
    tokenizer = Tokenizer()
    tok_path = os.path.join(config.DATA_DIR, "tokenizer.json")
    if os.path.exists(tok_path): tokenizer.load(tok_path)
    else: tokenizer.build_from_texts(["".join(p) for p in poems]); tokenizer.save(tok_path)
    train_loader, val_loader = create_dataloaders(poems, tokenizer, config.BATCH_SIZE)

    from models.lstm import CharLSTM
    from models.transformer import CharTransformer
    model = (CharLSTM if is_lstm else CharTransformer)(tokenizer.vocab_size).to(device)
    print(f"参数量: {model.count_parameters():,} ({model.count_parameters()/1e6:.2f}M)")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.LR_STEP,
                                                  gamma=config.LR_GAMMA)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f'{model_name}_best.pt')
    start_epoch, history, best_ppl, patience_counter = 1, [], float('inf'), 0
    patience = config.EARLY_STOP_PATIENCE

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if hasattr(model, 'tie_weights'): model.tie_weights()
        try: optimizer.load_state_dict(ckpt.get('optimizer_state_dict', {}))
        except: pass
        start_epoch = ckpt.get('epoch', 0) + 1
        history = ckpt.get('history', []); best_ppl = ckpt.get('val_ppl', float('inf'))
        if best_ppl != float('inf') and history:
            patience_counter = sum(1 for h in history[-(patience+1):]
                                   if h['val_ppl'] >= best_ppl * (1 - config.EARLY_STOP_MIN_DELTA))
        print(f"续训 epoch {start_epoch} (best PPL={best_ppl:.1f})")

    t0 = time.time()
    for epoch in range(start_epoch, config.EPOCHS + 1):
        train_ppl = train.train_epoch(model, train_loader, optimizer, device, is_lstm)
        val_ppl = train.validate(model, val_loader, device, is_lstm)
        scheduler.step()
        history.append({'epoch': epoch, 'train_ppl': round(train_ppl,2),
                        'val_ppl': round(val_ppl,2), 'lr': optimizer.param_groups[0]['lr']})
        elapsed = time.time() - t0
        improved = best_ppl == float('inf') or \
                   (best_ppl - val_ppl) / best_ppl >= config.EARLY_STOP_MIN_DELTA
        flag = " *" if improved else f" ({patience_counter+1}/{patience})"
        print(f"E{epoch:2d} | Train PPL:{train_ppl:.0f} Val PPL:{val_ppl:.0f}{flag} "
              f"| {elapsed/60:.1f}m")
        if improved:
            best_ppl = val_ppl; patience_counter = 0
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'tokenizer_path': tok_path, 'vocab_size': tokenizer.vocab_size,
                        'epoch': epoch, 'val_ppl': val_ppl, 'history': history}, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停! best PPL={best_ppl:.0f}"); break

    hist_path = os.path.join(config.CHECKPOINT_DIR, f'{model_name}_history.json')
    with open(hist_path, 'w') as f: json.dump(history, f, indent=2)
    train.plot_curves(history, os.path.join(config.CHECKPOINT_DIR, f'{model_name}_curves.png'))
    print(f"训练完成! best Val PPL={best_ppl:.0f}")
    return best_ppl

# ═══════════════ 分析 ═══════════════

def analyze_per_sample(results, mode, model_name):
    """对5组测试输入的生成结果逐组分析: 跑题/重复/结构"""
    inputs = config.FIRST_LINES if mode == 'first_line' else config.ACROSTICS
    per = config.GEN_NUM
    divider(f"{model_name} {'首句续写' if mode == 'first_line' else '藏头诗'} 逐组分析")

    for idx, inp in enumerate(inputs):
        g = results[idx * per: (idx + 1) * per]
        fmt_ok = sum(1 for r in g if len(r['lines']) == 4 and all(len(l) == 7 for l in r['lines']))
        avg_rel = sum(r.get('relevance', 0) or 0 for r in g) / len(g)
        reps = [compute_self_repeat(r['lines']) for r in g]
        avg_bg = sum(r['bigram'] for r in reps) / len(reps)
        avg_tg = sum(r['trigram'] for r in reps) / len(reps)
        avg_ol = sum(r['max_overlap'] for r in reps) / len(reps)

        issues = []
        if fmt_ok < per: issues.append(f"⚠ 格式不全 {fmt_ok}/{per}")
        if mode == 'first_line':
            issues.append(f"{'⚠ 跑题风险' if avg_rel < 0.8 else '✅ 主题紧凑' if avg_rel >= 1.0 else '  基本不跑题'} (rel={avg_rel:.3f})")
        if avg_bg > 0.40 or avg_tg > 0.25 or avg_ol > 0.55:
            issues.append(f"❌ 重复: bg={avg_bg:.2f} tg={avg_tg:.2f} ol={avg_ol:.2f}")
        elif avg_bg < 0.20 and avg_tg < 0.10:
            issues.append(f"✅ 低重复 (bg={avg_bg:.2f} tg={avg_tg:.2f})")
        else:
            issues.append(f"  重复正常 (bg={avg_bg:.2f} tg={avg_tg:.2f})")
        status = "✅" if all(not s.startswith('❌') and not s.startswith('⚠') for s in issues) else ""
        print(f"\n  [{idx+1}/5] {inp if mode == 'first_line' else '·'.join(inp)} {status}")
        for s in issues: print(f"    {s}")
        for i, r in enumerate(g): print_poem(r, i+1, mode)

    n = len(results)
    fmt_t = sum(1 for r in results if len(r['lines']) == 4 and all(len(l) == 7 for l in r['lines']))
    print(f"\n  ▶ 合计{n}首, 格式合规{fmt_t}/{n}" +
          (f", 均相关度{sum(r.get('relevance',0)or 0 for r in results)/n:.3f}" if mode == 'first_line' else ""))

def compare_sampling_params():
    """不同 T/top_k/rep_penalty 生成对比"""
    divider("采样策略参数对比")
    test_line = config.FIRST_LINES[0]
    print(f"\n  测试: 「{test_line}」\n")
    configs = [
        ("贪心 T=0                 ", 0.0, 0,  1.0, 0),
        ("T=0.3 top_k=10 rp=1.0   ", 0.3, 10, 1.0, 0),
        ("T=0.6 top_k=30 rp=1.10  ", 0.6, 30, 1.10, 0),
        ("T=0.8 top_k=40 rp=1.15 ★", 0.8, 40, 1.15, 3),
        ("T=1.0 top_k=60 rp=1.20  ", 1.0, 60, 1.20, 0),
        ("T=1.4 top_k=80 rp=1.25  ", 1.4, 80, 1.25, 0),
    ]
    try:
        model, tokenizer = load_model('transformer')
    except FileNotFoundError:
        print("  ⚠ 未找到 Transformer checkpoint"); return

    print(f"  {'策略':<30} {'rel':>7} {'bg↓':>7} {'tg↓':>7} {'ol↓':>7}")
    print(f"  {'─'*30} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
    for label, T, K, RP, NGR in configs:
        s = Sampler(temperature=T, top_k=K, repetition_penalty=RP, no_repeat_ngram_size=NGR)
        lines = generate_first_line(model, tokenizer, s, test_line)
        rep = compute_self_repeat(lines)
        rel = compute_relevance(test_line, lines[1:4] if len(lines) >= 4 else lines, 'transformer')
        print(f"  {label:<30} {rel:>7.3f} {rep['bigram']:>7.3f} "
              f"{rep['trigram']:>7.3f} {rep['max_overlap']:>7.3f}")
        print(f"    {' / '.join(lines)}")
    print(f"\n  推荐: T=0.6~1.0, top_k=30~60, rp=1.10~1.20, ngram=3")

# ═══════════════ 保存全部诗歌 ═══════════════

def save_all_poems(models_data, sample_dir):
    """将所有生成的诗保存到 outputs/samples/*.txt"""
    os.makedirs(sample_dir, exist_ok=True)
    for mn, fl, ac in models_data:
        for mode, data, label in [('first_line', fl, '首句续写'), ('acrostic', ac, '藏头诗')]:
            path = os.path.join(sample_dir, f'{mn}_{mode}.txt')
            inputs = config.FIRST_LINES if mode == 'first_line' else config.ACROSTICS
            with open(path, 'w', encoding='utf-8') as f:
                f.write(f"{'='*50}\n{mn.upper()} {label}\n{'='*50}\n\n")
                for idx, inp in enumerate(inputs):
                    f.write(f"【{idx+1}/5】输入: {inp if mode == 'first_line' else '·'.join(inp)}\n")
                    for i in range(config.GEN_NUM):
                        r = data[idx * config.GEN_NUM + i]
                        rep = compute_self_repeat(r['lines'])
                        rel = r.get('relevance', 0)
                        f.write(f"  #{i+1} rel:{rel:.3f} bg:{rep['bigram']:.2f} "
                                f"tg:{rep['trigram']:.2f} ol:{rep['max_overlap']:.2f}\n")
                        for l in r['lines']: f.write(f"    {l}\n")
                    f.write("\n")
            print(f"  ✓ {path}")

# ═══════════════ 主程序 ═══════════════

def main():
    t_total = time.time()

    if config.MODE == 'train':
        run_training(config.MODEL)
    elif config.MODE == 'generate':
        mn = config.MODEL
        fl, ac = run_generation(mn)
        divider(f"首句续写 ({mn}) 每输入1首")
        for idx in range(len(config.FIRST_LINES)): print_poem(fl[idx * config.GEN_NUM], idx+1, 'first_line')
        divider(f"藏头诗 ({mn}) 每输入1首")
        for idx in range(len(config.ACROSTICS)): print_poem(ac[idx * config.GEN_NUM], idx+1, 'acrostic')
        save_all_poems([(mn, fl, ac)], config.SAMPLE_DIR)
    elif config.MODE == 'eval':
        import eval; eval.main()
    elif config.MODE == 'full':
        # 1. 训练
        lstm_ppl = run_training('lstm')
        trans_ppl = run_training('transformer')
        ft_loss = train.finetune_rhyme_tone()

        # 2. 生成
        lstm_fl, lstm_ac = run_generation('lstm')
        trans_fl, trans_ac = run_generation('transformer')
        rhyme_fl, rhyme_ac = run_generation('rhyme_transformer')

        # 3. 样例 (每种只展示1首)
        for mn, fl, ac in [('LSTM', lstm_fl, lstm_ac), ('Transformer', trans_fl, trans_ac),
                            ('RhymeTone', rhyme_fl, rhyme_ac)]:
            divider(f"{mn} 首句续写样例 (每输入1首)")
            for idx in range(len(config.FIRST_LINES)): print_poem(fl[idx * config.GEN_NUM], idx+1, 'first_line')
            divider(f"{mn} 藏头诗样例 (每输入1首)")
            for idx in range(len(config.ACROSTICS)): print_poem(ac[idx * config.GEN_NUM], idx+1, 'acrostic')

        # 4. 指标
        lm_fl = compute_all_metrics(lstm_fl); lm_ac = compute_all_metrics(lstm_ac)
        tm_fl = compute_all_metrics(trans_fl); tm_ac = compute_all_metrics(trans_ac)
        rm_fl = compute_all_metrics(rhyme_fl); rm_ac = compute_all_metrics(rhyme_ac)

        divider("评测报告")
        print(f"\n  训练: LSTM PPL={lstm_ppl:.0f}  Transformer PPL={trans_ppl:.0f}  "
              f"RhymeTone Loss={ft_loss:.4f}")
        print_metrics_table({'lstm': lm_fl, 'transformer': tm_fl, 'rhyme_t': rm_fl}, "首句续写 (各25首)")
        print_metrics_table({'lstm': lm_ac, 'transformer': tm_ac, 'rhyme_t': rm_ac}, "藏头诗 (各25首)")

        # 5. 韵脚诊断
        debug_rhyme(trans_fl[:10], "Transformer 首句续写 韵脚诊断")

        # 6. 逐样本分析
        for mn, fl, ac in [('LSTM', lstm_fl, lstm_ac), ('Transformer', trans_fl, trans_ac),
                            ('RhymeTone', rhyme_fl, rhyme_ac)]:
            analyze_per_sample(fl, 'first_line', mn)
            analyze_per_sample(ac, 'acrostic', mn)

        # 7. 采样对比
        compare_sampling_params()

        # 8. 保存全部诗歌 → outputs/samples/*.txt
        divider("保存诗歌")
        models_data = [('lstm', lstm_fl, lstm_ac), ('transformer', trans_fl, trans_ac),
                       ('rhyme', rhyme_fl, rhyme_ac)]
        save_all_poems(models_data, config.SAMPLE_DIR)

        # 9. 保存 JSON 报告
        report = {'lstm_val_ppl': lstm_ppl, 'transformer_val_ppl': trans_ppl,
                  'rhyme_transformer_ft_loss': ft_loss,
                  'first_line': {'lstm': lm_fl, 'transformer': tm_fl, 'rhyme_transformer': rm_fl},
                  'acrostic': {'lstm': lm_ac, 'transformer': tm_ac, 'rhyme_transformer': rm_ac}}
        os.makedirs(config.SAMPLE_DIR, exist_ok=True)
        with open(os.path.join(config.SAMPLE_DIR, 'full_report.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n报告已保存 | 总耗时 {(time.time()-t_total)/60:.1f}m")
    else:
        print(f"未知 MODE: {config.MODE}")

if __name__ == '__main__':
    main()
