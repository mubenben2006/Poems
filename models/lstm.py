"""
字符级 LSTM 语言模型
架构: Embedding → 2-layer LSTM → Linear → vocab_size
"""
import torch
import torch.nn as nn
import config


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=None, hidden_dim=None,
                 num_layers=None, dropout=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim or config.EMBED_DIM
        self.hidden_dim = hidden_dim or config.LSTM_HIDDEN
        self.num_layers = num_layers or config.LSTM_LAYERS
        self.dropout_rate = dropout or config.DROPOUT

        self.embedding = nn.Embedding(vocab_size, self.embed_dim,
                                      padding_idx=config.PAD_IDX)
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim, self.num_layers,
                            dropout=self.dropout_rate if self.num_layers > 1 else 0,
                            batch_first=True)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.output = nn.Linear(self.hidden_dim, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)  # forget gate bias = 1

    def forward(self, x, hidden=None):
        emb = self.dropout(self.embedding(x))
        lstm_out, hidden = self.lstm(emb, hidden)
        logits = self.output(self.dropout(lstm_out))
        return logits, hidden

    def init_hidden(self, batch_size=1, device=None):
        device = device or config.DEVICE
        shape = (self.num_layers, batch_size, self.hidden_dim)
        return (torch.zeros(shape, device=device),
                torch.zeros(shape, device=device))

    def generate_step(self, token, hidden, sampler, generated_ids=None,
                       suppress_tokens=None):
        """单步生成: 输入 token + hidden → 输出 next_token + new_hidden"""
        x = torch.tensor([[token]], device=config.DEVICE)
        logits, hidden = self.forward(x, hidden)
        return sampler.sample(logits[0, -1],
                              generated_ids=generated_ids,
                              suppress_tokens=suppress_tokens), hidden

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
