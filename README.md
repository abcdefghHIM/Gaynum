# 基于模型的指数生成器

```python
def collate_fn(batch):
    # batch = [(x_t, y_t), (x_t, y_t), ...]
    xs, ys = zip(*batch)

    # padding
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0)  # (B, T)
    ys = torch.stack(ys)

    key_padding_mask = xs_padded == 0

    return xs_padded, ys, key_padding_mask

class TextDataset(Dataset):
    def __init__(self, txt_path, tokenizer, max_length=32):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(txt_path, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        left, right = self.lines[idx].rsplit(",", 1)
        text = left
        if len(text) > 2:
            ok = random.randint(0, 1)
            if ok:
                length = random.randint(2, len(text))
                start = random.randint(0, len(text) - length)
                cropped = text[start : start + length]
            else:
                cropped = text
        else:
            cropped = text
        score = float(right)
        x = tokenizer.encode(cropped, add_special_tokens=False)
        if len(x) > self.max_length:
            x = x[: self.max_length]
        x_t = torch.tensor(x, dtype=torch.long)
        y_t = torch.tensor(score, dtype=torch.float)
        return x_t, y_t

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, head_dim):
        super().__init__()
        self.W_q = nn.Linear(d_model, head_dim, bias=False)
        self.W_k = nn.Linear(d_model, head_dim, bias=False)
        self.W_v = nn.Linear(d_model, head_dim, bias=False)

    def forward(self, x, key_padding_mask=None):
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask[:, None, :], float("-inf")
            )
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        return out, attn_weights

class GayModel(nn.Module):
    def __init__(self, vocab_size, d_model=16, head_dim=16, hidden_dim=16):
        super(GayModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attn = SingleHeadAttention(d_model, head_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(head_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, key_padding_mask=None):
        emb = self.embedding(x)
        attn_out, weights = self.attn(emb, key_padding_mask)
        attn_out += emb
        pooled = self.pool(attn_out.transpose(1, 2)).squeeze(-1)
        h = F.silu(self.fc1(pooled))
        score = F.silu(self.fc2(h))
        score = torch.sigmoid(self.fc3(h)).squeeze(-1)
        return score
```
