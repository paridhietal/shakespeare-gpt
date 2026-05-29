import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gradio as gr

# ── Model Definition ──────────────────────────────────────────

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, block_size, dropout=0.1):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_k, bias=False)
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, X):
        B, T, _ = X.shape
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return weights @ V

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.heads = nn.ModuleList([
            SingleHeadAttention(d_model, self.d_k, block_size, dropout)
            for _ in range(num_heads)
        ])
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X):
        head_outputs = [head(X) for head in self.heads]
        concat = torch.cat(head_outputs, dim=-1)
        return self.W_O(concat)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, block_size, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, block_size, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, X):
        attended = self.attention(X)
        X = self.norm1(X + attended)
        fed_forward = self.ff(X)
        X = self.norm2(X + fed_forward)
        return X

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=100):
        super().__init__()
        PE = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        PE[:, 0::2] = torch.sin(position * div_term)
        PE[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('PE', PE)

    def forward(self, X):
        seq_len = X.shape[1]
        return X + self.PE[:seq_len, :]

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len=block_size)
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, num_heads, block_size)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, targets=None):
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.output_head(x)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

# ── Load Model ────────────────────────────────────────────────

BLOCK_SIZE = 64
VOCAB = sorted(set(open('shakespeare_gpt_vocab.txt').read()))
char_to_idx = {ch: i for i, ch in enumerate(VOCAB)}
idx_to_char = {i: ch for i, ch in enumerate(VOCAB)}
encode = lambda s: [char_to_idx.get(c, 0) for c in s]
decode = lambda l: ''.join([idx_to_char[i] for i in l])

device = 'cpu'  # HuggingFace free tier is CPU

model = GPT(
    vocab_size  = len(VOCAB),
    d_model     = 128,
    num_heads   = 4,
    num_layers  = 4,
    block_size  = BLOCK_SIZE
).to(device)

model.load_state_dict(torch.load('shakespeare_gpt.pth', map_location=device))
model.eval()

# ── Generate Function ─────────────────────────────────────────

def generate(start_text, temperature, max_new_tokens):
    tokens = torch.tensor(encode(start_text), dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            tokens_crop = tokens[:, -BLOCK_SIZE:]
            logits, _ = model(tokens_crop)
            logits_last = logits[:, -1, :] / temperature
            probs = F.softmax(logits_last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
    return decode(tokens[0].tolist())

# ── Gradio Interface ──────────────────────────────────────────

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(value="ROMEO: ", label="Start Text"),
        gr.Slider(minimum=0.5, maximum=1.5, value=0.6, step=0.1, label="Temperature"),
        gr.Slider(minimum=50, maximum=500, value=200, step=50, label="Max Tokens"),
    ],
    outputs=gr.Textbox(label="Generated Shakespeare"),
    title="🎭 Shakespeare GPT",
    description="A GPT trained from scratch on Shakespeare. Built with pure PyTorch — no HuggingFace models used.",
    examples=[
        ["ROMEO: ", 0.6, 200],
        ["HAMLET: To be or not to be", 0.7, 300],
        ["KING RICHARD: ", 0.6, 200],
    ]
)

demo.launch()
