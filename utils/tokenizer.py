"""字符级分词器 —— 特殊 token 作为原子单位处理"""
import json
from collections import Counter
import config


class Tokenizer:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0
        self._built = False

    def build_from_texts(self, texts):
        """统计字频，构建词表"""
        counter = Counter()
        for text in texts:
            for ch in text:
                counter[ch] += 1

        sorted_chars = sorted(counter.items(), key=lambda x: -x[1])
        vocab = list(config.SPECIAL_TOKENS)

        for ch, freq in sorted_chars:
            if len(vocab) >= config.MAX_VOCAB_SIZE:
                break
            if freq >= config.MIN_CHAR_FREQ and ch not in config.SPECIAL_TOKENS:
                vocab.append(ch)

        self.char2idx = {ch: i for i, ch in enumerate(vocab)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        self.vocab_size = len(vocab)
        self._built = True
        print(f"词表: {self.vocab_size} tokens")

    def encode(self, text):
        """
        字符串 → token ID 列表。
        特殊 token（如 <BOS>）作为原子单位匹配，不被拆成单字符。
        """
        result = []
        i = 0
        specials = sorted(config.SPECIAL_TOKENS, key=len, reverse=True)  # 长优先
        while i < len(text):
            matched = False
            for sp in specials:
                if text[i:i+len(sp)] == sp:
                    result.append(self.char2idx[sp])
                    i += len(sp)
                    matched = True
                    break
            if not matched:
                ch = text[i]
                result.append(self.char2idx.get(ch, config.UNK_IDX))
                i += 1
        return result

    def decode(self, indices, skip_special=True):
        """token ID 列表 → 字符串"""
        special = {config.PAD_IDX, config.BOS_IDX, config.EOS_IDX,
                   config.SEP_IDX, config.UNK_IDX}
        chars = []
        for idx in indices:
            if skip_special and idx in special:
                continue
            chars.append(self.idx2char.get(idx, config.UNK_TOKEN))
        return "".join(chars)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "char2idx": self.char2idx,
                "idx2char": {str(k): v for k, v in self.idx2char.items()},
                "vocab_size": self.vocab_size,
            }, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.char2idx = data["char2idx"]
        self.idx2char = {int(k): v for k, v in data["idx2char"].items()}
        self.vocab_size = data["vocab_size"]
        self._built = True
