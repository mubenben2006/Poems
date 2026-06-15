"""
诗歌生成 & 评测
用法:
  python generate.py                          默认: 全量生成 + 指标对比
  python generate.py -m lstm -i "爆竹声中一岁除"    首句续写
  python generate.py -m transformer --acrostic "梅兰竹菊" 藏头诗
"""
import os, sys, re, json
import torch
import config
from utils.tokenizer import Tokenizer
from utils.sampler import Sampler
from utils.metrics import (compute_self_repeat, compute_relevance,
                            compute_format_rate, compute_rhyme_rate,
                            compute_tone_rate, find_top_poems)
from models.lstm import CharLSTM
from models.transformer import CharTransformer
from models.rhyme_transformer import RhymeToneTransformer

# ═══════════════ 模型加载 ═══════════════

def load_model(model_name):
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f'{model_name}_best.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f'{model_name}_final.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 {model_name} checkpoint: {ckpt_path}\n请先训练")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    tokenizer = Tokenizer()
    tok_path = ckpt.get('tokenizer_path', '')
    if not os.path.exists(tok_path):
        tok_path = os.path.join(config.DATA_DIR, 'tokenizer.json')
    tokenizer.load(tok_path)

    if model_name == 'lstm':
        model = CharLSTM(ckpt['vocab_size'])
    elif model_name == 'transformer':
        model = CharTransformer(ckpt['vocab_size'])
    elif model_name == 'rhyme_transformer':
        n_rc = ckpt.get('num_rhyme_classes', 40)
        model = RhymeToneTransformer(ckpt['vocab_size'], n_rc)
    else:
        raise ValueError(f"未知模型: {model_name}")
    model.load_state_dict(ckpt['model_state_dict'])
    if hasattr(model, 'tie_weights'): model.tie_weights()
    if model_name == 'rhyme_transformer': model.transformer.tie_weights()

    model.to(config.DEVICE); model.eval()
    info = f"epoch={ckpt.get('epoch','?')}"
    if 'val_ppl' in ckpt: info += f", Val PPL={ckpt['val_ppl']:.1f}"
    elif 'combined_loss' in ckpt: info += f", FT Loss={ckpt['combined_loss']:.4f}"
    print(f"加载 {model_name} ({info})")
    return model, tokenizer

# ═══════════════ 生成核心 ═══════════════

def _gen_one_line(model, tokenizer, sampler, prompt, is_lstm, max_new=7):
    """生成一句: 直接模型采样 7 个汉字, SEP/EOS 全程屏蔽"""
    prompt_ids = tokenizer.encode(prompt)
    gen = list(prompt_ids)
    n = len(prompt_ids)
    suppress = {config.SEP_IDX, config.EOS_IDX}

    with torch.no_grad():
        if is_lstm:
            hidden = model.init_hidden(1, config.DEVICE)
            for t in prompt_ids:
                _, hidden = model(torch.tensor([[t]], device=config.DEVICE), hidden)
            t = prompt_ids[-1]
            for _ in range(max_new):
                t, hidden = model.generate_step(t, hidden, sampler,
                                                 generated_ids=gen[n:],
                                                 suppress_tokens=suppress)
                gen.append(t)
        else:
            x = torch.tensor([prompt_ids], device=config.DEVICE)
            logits, pkv = model(x, use_cache=True)
            t = sampler.sample(logits[0, -1], suppress_tokens=suppress)
            gen.append(t)
            for _ in range(max_new - 1):
                x = torch.tensor([[t]], device=config.DEVICE)
                logits, pkv = model(x, past_kvs=pkv, use_cache=True)
                t = sampler.sample(logits[0, -1], generated_ids=gen[n:],
                                    suppress_tokens=suppress)
                gen.append(t)

    new_text = tokenizer.decode(gen[n:], skip_special=True)
    return ''.join(re.findall(r'[一-鿿]', new_text))[:max_new]

def generate_first_line(model, tokenizer, sampler, first_line):
    """首句续写: 逐句生成, 每句见前句末字 → 自然押韵"""
    is_lstm = isinstance(model, CharLSTM)
    first_line = ''.join(re.findall(r'[一-鿿]', first_line)[:7]).ljust(7, '□')
    lines = [first_line]
    ctx = config.BOS_TOKEN + first_line

    for i in range(3):  # 生成后三句
        prompt = ctx + config.SEP_TOKEN
        new = _gen_one_line(model, tokenizer, sampler, prompt, is_lstm)
        lines.append(new.ljust(7, '□'))
        ctx += config.SEP_TOKEN + new
    return lines

def generate_acrostic(model, tokenizer, sampler, chars):
    """藏头诗: 逐句生成, 前句末字可见 → 自然押韵 (复用 _gen_one_line)"""
    is_lstm = isinstance(model, CharLSTM)
    chars = (re.findall(r'[一-鿿]', chars)[:4] + ['诗'] * 4)[:4]
    lines, ctx = [], ''

    for i, ch in enumerate(chars):
        if i == 0:
            prompt = config.BOS_TOKEN + ch
        else:
            ctx += lines[-1] + config.SEP_TOKEN
            prompt = config.BOS_TOKEN + ctx + ch

        new = _gen_one_line(model, tokenizer, sampler, prompt, is_lstm)
        line = ch + new[:6]
        lines.append(line.ljust(7, '□'))
    return lines

# ═══════════════ 批量生成 (供 run.py 复用) ═══════════════

def run_generation(model_name, first_lines=None, acrostics=None):
    """加载模型并批量生成，返回 (first_results, acro_results)"""
    first_lines = first_lines or config.FIRST_LINES
    acrostics = acrostics or config.ACROSTICS
    model, tokenizer = load_model(model_name)
    sampler = Sampler(temperature=config.TEMPERATURE, top_k=config.TOP_K,
                       top_p=config.TOP_P,
                       repetition_penalty=config.REPETITION_PENALTY,
                       no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE)
    fl, ac = [], []
    for inp in first_lines:
        for _ in range(config.GEN_NUM):
            lines = generate_first_line(model, tokenizer, sampler, inp)
            rel = compute_relevance(inp, lines[1:4] if len(lines) >= 4 else lines, model_name)
            fl.append({"lines": lines, "relevance": rel, "input": inp})
    for inp in acrostics:
        for _ in range(config.GEN_NUM):
            lines = generate_acrostic(model, tokenizer, sampler, inp)
            rel = compute_relevance(''.join(inp), lines, model_name)
            ac.append({"lines": lines, "relevance": rel, "input": inp})
    return fl, ac

# ═══════════════ 指标 (单一来源) ═══════════════

def compute_all_metrics(results):
    fmt = compute_format_rate(results)
    rhyme = compute_rhyme_rate(results)
    tone = compute_tone_rate(results)
    bg = tg = ol = rel_sum = n = 0.0
    for r in results:
        lines = r['lines']
        if len(lines) == 4 and all(len(l) == 7 for l in lines):
            rep = compute_self_repeat(lines)
            bg += rep['bigram']; tg += rep['trigram']; ol += rep['max_overlap']
            if r.get('relevance') is not None: rel_sum += r['relevance']
            n += 1
    n = max(n, 1)
    return {'count': int(len(results)), 'format': float(fmt), 'rhyme': float(rhyme),
            'tone': float(tone), 'relevance': float(rel_sum / n),
            'bigram': float(bg / n), 'trigram': float(tg / n), 'overlap': float(ol / n)}

# ═══════════════ 输出 ═══════════════

def divider(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def print_poem(r, idx, mode):
    """紧凑打印一首诗"""
    lines, rep = r['lines'], compute_self_repeat(r['lines'])
    inp = r.get('input', '')
    rel = r.get('relevance', 0)
    tag = f"[{inp}]" if mode == 'first_line' else f"[{'·'.join(inp)}]"
    print(f"  {tag} #{idx} rel:{rel:.3f} bg:{rep['bigram']:.2f} "
          f"tg:{rep['trigram']:.2f} ol:{rep['max_overlap']:.2f}")
    for l in lines: print(f"    {l}")

def print_metrics_table(metrics_dict, title):
    models = list(metrics_dict.keys())
    print(f"\n  {title}")
    print(f"  {'指标':<16}" + "".join(f" {m.upper():>12}" for m in models))
    print("  " + "-" * (16 + 13 * len(models)))
    rows = [("样本数", 'count', 'd'), ("格式合规率", 'format', '.1%'),
            ("押韵率", 'rhyme', '.1%'), ("平仄匹配率", 'tone', '.1%'),
            ("语义相关度", 'relevance', '.3f'), ("Bigram重复↓", 'bigram', '.4f'),
            ("Trigram重复↓", 'trigram', '.4f'), ("行间最大重叠↓", 'overlap', '.4f')]
    for label, key, fmt in rows:
        vals = []
        for m in models:
            v = metrics_dict[m].get(key)
            vals.append(f"{v:>12d}" if fmt == 'd' and v is not None
                        else (f"{v:{fmt}}" if v is not None else "N/A"))
        print(f"  {label:<16}" + "".join(f" {v:>12}" for v in vals))

# ═══════════════ 主程序 ═══════════════

def main():
    if len(sys.argv) > 1:
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument('-m', '--model', default='lstm',
                       choices=['lstm', 'transformer', 'rhyme_transformer'])
        p.add_argument('-i', '--input', default=None)
        p.add_argument('--acrostic', default=None)
        p.add_argument('-n', '--num', type=int, default=config.GEN_NUM)
        p.add_argument('--T', type=float, default=config.TEMPERATURE)
        p.add_argument('--top_k', type=int, default=config.TOP_K)
        args = p.parse_args()
        sampler = Sampler(temperature=args.T, top_k=args.top_k,
                           repetition_penalty=config.REPETITION_PENALTY,
                           no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE)
        model, tokenizer = load_model(args.model)
        if args.acrostic:
            print(f"\n藏头: 「{'」「'.join(args.acrostic)}」 (T={args.T}, top_k={args.top_k})\n")
            for i in range(args.num):
                lines = generate_acrostic(model, tokenizer, sampler, args.acrostic)
                rep = compute_self_repeat(lines)
                ok = all(j < len(lines) and lines[j] and lines[j][0] == args.acrostic[j]
                        for j in range(min(4, len(lines))))
                print(f"--- {i+1} ---\n" + "\n".join(lines))
                print(f"bg:{rep['bigram']:.2f} tg:{rep['trigram']:.2f} "
                      f"ol:{rep['max_overlap']:.2f} 藏头:{'✓' if ok else '✗'}\n")
        else:
            fl = (args.input or config.GEN_INPUT).strip()
            print(f"\n续写: 「{fl}」 (T={args.T}, top_k={args.top_k})\n")
            for i in range(args.num):
                lines = generate_first_line(model, tokenizer, sampler, fl)
                rep = compute_self_repeat(lines)
                rel = compute_relevance(fl, lines[1:4] if len(lines) >= 4 else lines, args.model)
                print(f"--- {i+1} ---\n" + "\n".join(lines))
                print(f"rel:{rel:.3f} bg:{rep['bigram']:.2f} tg:{rep['trigram']:.2f} "
                      f"ol:{rep['max_overlap']:.2f}\n")
        return

    # 默认: 全量双模型 → 指标对比
    sampler = Sampler(temperature=config.TEMPERATURE, top_k=config.TOP_K,
                       top_p=config.TOP_P,
                       repetition_penalty=config.REPETITION_PENALTY,
                       no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE)
    models_ok = ['lstm', 'transformer', 'rhyme_transformer']
    all_fl, all_ac = {}, {}
    for mn in models_ok:
        try: fl, ac = run_generation(mn); all_fl[mn] = fl; all_ac[mn] = ac
        except FileNotFoundError as e: print(f"\n  ⚠ {e}")
    if not all_fl: print("\n无可用的模型"); return

    for mn in all_fl:
        for _ in range(min(3, len(all_fl[mn]))):
            print_poem(all_fl[mn][_], _+1, 'first_line')
        for _ in range(min(3, len(all_ac[mn]))):
            print_poem(all_ac[mn][_], _+1, 'acrostic')

    fl_m = {mn: compute_all_metrics(all_fl[mn]) for mn in all_fl}
    ac_m = {mn: compute_all_metrics(all_ac[mn]) for mn in all_ac}
    print_metrics_table(fl_m, "首句续写 (各25首)")
    print_metrics_table(ac_m, "藏头诗 (各25首)")

    # 质量 Top-5
    divider("质量 Top-5")
    top = find_top_poems([(mn, all_fl[mn] + all_ac[mn]) for mn in all_fl], top_n=5)
    for rank, (score, mn, r) in enumerate(top, 1):
        inp = r.get('input', '')
        mode = '续' if len(inp) == 7 or (isinstance(inp, str) and len(re.findall(r'[一-鿿]', inp)) <= 7) else '藏'
        print(f"  #{rank} [{mn}] score={score:.2f} | {inp}")
        for l in r['lines']: print(f"    {l}")
        print()

    os.makedirs(config.SAMPLE_DIR, exist_ok=True)
    with open(os.path.join(config.SAMPLE_DIR, 'generation_report.json'), 'w', encoding='utf-8') as f:
        json.dump({'first_line': fl_m, 'acrostic': ac_m}, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存")

if __name__ == '__main__':
    main()
