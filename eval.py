"""
评测脚本 —— PPL / 韵脚诊断 / 采样对比（生成+指标复用 generate.py）
用法: python eval.py [--model rhyme_transformer] [--num_samples 25]
"""
import os, sys, json
import torch
import config
from utils.tokenizer import Tokenizer
from utils.data_loader import prepare_data, create_dataloaders
from utils.metrics import compute_ppl, debug_rhyme, compute_self_repeat, compute_relevance
from utils.sampler import Sampler
from generate import (load_model, generate_first_line, generate_acrostic,
                       compute_all_metrics, divider, print_poem, print_metrics_table)


def main():
    # ── CLI ──
    if len(sys.argv) > 1:
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument('--model', default=config.MODEL,
                       choices=['lstm', 'transformer', 'rhyme_transformer'])
        p.add_argument('--num_samples', type=int, default=config.GEN_NUM)
        p.add_argument('--T', type=float, default=config.TEMPERATURE)
        p.add_argument('--top_k', type=int, default=config.TOP_K)
        args = p.parse_args()
        mn, ns, T, K = args.model, args.num_samples, args.T, args.top_k
    else:
        mn, ns, T, K = config.MODEL, config.GEN_NUM, config.TEMPERATURE, config.TOP_K

    device = config.DEVICE
    print(f"模型: {mn} | 设备: {device}")

    # ═══════════════ 1. PPL ═══════════════
    model, tokenizer = load_model(mn)
    model.to(device)
    poems = prepare_data()
    _, val_loader = create_dataloaders(poems, tokenizer, batch_size=64)
    ppl = compute_ppl(model, val_loader, device)
    print(f"Val PPL: {ppl:.2f}")

    # ═══════════════ 2. 生成 + 指标 ═══════════════
    sampler = Sampler(temperature=T, top_k=K, top_p=config.TOP_P,
                       repetition_penalty=config.REPETITION_PENALTY,
                       no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE)

    fl_results, ac_results = [], []
    for inp in config.FIRST_LINES:
        for _ in range(max(1, ns // len(config.FIRST_LINES))):
            lines = generate_first_line(model, tokenizer, sampler, inp)
            rel = compute_relevance(inp, lines[1:4] if len(lines) >= 4 else lines, mn)
            fl_results.append({"lines": lines, "relevance": rel, "input": inp})

    for inp in config.ACROSTICS:
        for _ in range(max(1, ns // len(config.ACROSTICS))):
            lines = generate_acrostic(model, tokenizer, sampler, inp)
            ac_results.append({"lines": lines, "relevance": None, "input": inp})

    fl_m = compute_all_metrics(fl_results)
    ac_m = compute_all_metrics(ac_results)
    print_metrics_table({mn: fl_m}, "首句续写")
    print_metrics_table({mn: ac_m}, "藏头诗")

    # ═══════════════ 3. 韵脚诊断 ═══════════════
    debug_rhyme(fl_results[:10], f"{mn} 韵脚诊断")

    # ═══════════════ 4. 采样策略对比 ═══════════════
    divider("采样策略对比")
    test_line = config.FIRST_LINES[0]
    configs = [
        ("贪心 T=0            ", 0.0, 0,  1.0,  0),
        ("T=0.5 top_k=20 rp=1.10", 0.5, 20, 1.10, 0),
        ("T=0.8 top_k=40 rp=1.15", 0.8, 40, 1.15, 3),
        ("T=1.2 top_k=60 rp=1.20", 1.2, 60, 1.20, 0),
    ]
    for label, t, k, rp, ngr in configs:
        s = Sampler(temperature=t, top_k=k, repetition_penalty=rp,
                     no_repeat_ngram_size=ngr)
        lines = generate_first_line(model, tokenizer, s, test_line)
        rep = compute_self_repeat(lines)
        rel = compute_relevance(test_line, lines[1:4] if len(lines) >= 4 else lines, mn)
        print(f"\n  [{label}] rel:{rel:.3f}")
        print(f"  {' / '.join(lines)}")
        print(f"  bg:{rep['bigram']:.2f} tg:{rep['trigram']:.2f} ol:{rep['max_overlap']:.2f}")

    # ═══════════════ 5. 样例 ═══════════════
    divider("生成样例")
    for i, r in enumerate(fl_results[:5]):
        print_poem(r, i + 1, 'first_line')
    for i, r in enumerate(ac_results[:5]):
        print_poem(r, i + 1, 'acrostic')

    # ═══════════════ 6. 保存 ═══════════════
    os.makedirs(config.SAMPLE_DIR, exist_ok=True)
    with open(os.path.join(config.SAMPLE_DIR, f'{mn}_eval.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'ppl': float(ppl),
            'first_line': [{'lines': r['lines'], 'relevance': float(r['relevance']) if r['relevance'] is not None else None}
                           for r in fl_results],
            'acrostic': [{'lines': r['lines']} for r in ac_results],
        }, f, ensure_ascii=False, indent=2)
    print(f"样本已保存 → {config.SAMPLE_DIR}")


if __name__ == '__main__':
    main()
