"""Temperature + Top-K / Top-P + Repetition Penalty 采样"""
import torch
import torch.nn.functional as F


class Sampler:
    def __init__(self, temperature=1.0, top_k=0, top_p=0.0,
                 repetition_penalty=1.0, no_repeat_ngram_size=0):
        """
        Args:
            temperature:  温度缩放 (0 = 贪心, >0 = 随机)
            top_k:        Top-K 截断 (0 = 不限制)
            top_p:        Nucleus sampling 阈值 (0 = 不限制)
            repetition_penalty: 重复惩罚 (>1 惩罚已出现 token, 1 = 不惩罚)
            no_repeat_ngram_size: 禁止同一 n-gram 再次出现 (0 = 不限制, 建议 3)
        """
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size

    def sample(self, logits, generated_ids=None, suppress_tokens=None):
        """
        logits: (vocab_size,) or (1, vocab_size) — 原始 logits
        generated_ids: list[int] — 已生成的 token 序列, 用于重复惩罚
        suppress_tokens: set[int] | None — 强制屏蔽的 token (如行未满时屏蔽 SEP/EOS)
        """
        if logits.dim() == 2:
            logits = logits.squeeze(0)

        logits = logits.clone()  # 避免修改原始 tensor

        # ---- 1. 强制屏蔽 ----
        if suppress_tokens:
            for tid in suppress_tokens:
                logits[tid] = float('-inf')

        # ---- 2. 重复惩罚 (repetition_penalty) ----
        if self.repetition_penalty != 1.0 and generated_ids:
            for tid in set(generated_ids):
                if logits[tid] > 0:
                    logits[tid] /= self.repetition_penalty
                else:
                    logits[tid] *= self.repetition_penalty

        # ---- 3. n-gram 禁止 ----
        if self.no_repeat_ngram_size > 0 and generated_ids:
            n = self.no_repeat_ngram_size
            if len(generated_ids) >= n - 1:
                # 找到已出现的所有 n-gram，收集其下一个 token
                banned = set()
                for start in range(len(generated_ids) - n + 1):
                    ngram = tuple(generated_ids[start:start + n])
                    # 检查当前生成序列末尾 (n-1)-gram 是否匹配
                    if len(generated_ids) >= n - 1:
                        prefix = tuple(generated_ids[-(n - 1):])
                        full = tuple(generated_ids[start:start + n])
                        if len(full) >= n and full[:n-1] == prefix:
                            banned.add(full[n-1])
                for tid in banned:
                    logits[tid] = float('-inf')

        # ---- 4. Temperature ----
        if self.temperature > 0 and self.temperature != 1.0:
            logits = logits / self.temperature
        elif self.temperature == 0:
            return torch.argmax(logits).item()

        # ---- 5. Top-K ----
        if self.top_k > 0:
            k = min(self.top_k, (logits > float('-inf')).sum().item())
            if k > 0:
                threshold = torch.topk(logits, k, dim=-1).values[-1]
                logits[logits < threshold] = float('-inf')

        # ---- 6. Top-P (nucleus) ----
        if self.top_p > 0.0 and self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            remove = cumsum > self.top_p
            remove[..., 0] = False  # 至少保留一个
            remove = remove.scatter(0, sorted_indices, remove)
            logits[remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        if torch.isnan(probs).any() or probs.sum() == 0:
            return torch.argmax(logits).item()
        return torch.multinomial(probs, num_samples=1).item()
