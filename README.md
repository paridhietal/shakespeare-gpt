# shakespeare-gpt

# I Built GPT From Scratch. Here's What Happened.

No, not "I used the OpenAI API."
Not "I fine-tuned a HuggingFace model."

I mean I sat down, opened a blank Python file, and built the thing that powers ChatGPT — from absolute zero. Attention mechanism, positional encoding, transformer blocks, training loop. Every single line.

Then I fed it Shakespeare and watched it slowly lose its mind — and then slowly gain one.

**[See it live →](https://huggingface.co/spaces/manicc/shakespeare-gpt)**

---

## Why I did this

I kept seeing people put "Transformer models" on their resume after running `pipeline("text-generation")` in three lines.

I wanted to actually know what's happening inside. So I built it.

Turns out — it's a lot of matrix multiplications, a surprisingly elegant idea called attention, and a shocking amount of shape mismatch errors at 2am.

---

## What it does

You give it a prompt. It writes Shakespeare.

```
Input:   "HAMLET: "

Output:  "HAMLET: I have a violence, in this poor that an that cure
          of my rest to please them not the virtue...

          MENENIUS:
          I mean thee any what ever speak breats,
          By lies and such a words to be severity,
          Shall not in this traitor..."
```

Is the meaning coherent? Not really.
Does it sound disturbingly Shakespearean? Absolutely.

`MENENIUS` is a real character from Coriolanus. The model learned that — along with iambic structure, dramatic vocabulary, and verse formatting — purely by predicting one character at a time, 1 million times over.

That's it. That's the whole trick.

---

## What's actually inside

808,001 parameters. Built from scratch. Nothing borrowed.

| Component | What it does |
|---|---|
| Character tokenizer | Converts text → integers. No BPE, just raw chars. |
| Token embedding | Integers → learned 128-dim vectors |
| Positional encoding | Sine/cosine waves so the model knows word order |
| Multi-head attention | 4 heads, each learning different language patterns |
| Causal mask | Blocks the model from peeking at future tokens |
| Feed forward layers | Processes what attention gathered |
| Residual connections | Lets gradients flow deep without vanishing |
| Layer normalization | Keeps values stable through 4 stacked blocks |

No magic. Just these pieces, stacked carefully, trained long enough.

---

## Architecture

```
"HAMLET: "  →  character integers
                    ↓
            Token Embedding
          (integers → vectors)
                    ↓
         Positional Encoding
          (add position info)
                    ↓
        ┌─── Transformer Block ───┐
        │  Multi-Head Attention   │
        │  (4 heads, causal mask) │  × 4 layers
        │  Feed Forward Network   │
        │  Residual + LayerNorm   │
        └─────────────────────────┘
                    ↓
             Output Head
        (vectors → vocab scores)
                    ↓
          Next character: 'T'
```

---

## Training

```
Data:       1 million characters of Shakespeare
Split:      90% train / 10% validation  
Steps:      8,000
Optimizer:  AdamW
Batch:      32 sequences × 64 tokens
GPU:        Google Colab T4
```

Watching the loss drop in real time is genuinely one of the coolest things I've experienced:

```
Step     0 | train: 4.15 | val: 4.16  ← knows absolutely nothing
Step  1000 | train: 1.89 | val: 1.99  ← learned letters form words
Step  4000 | train: 1.52 | val: 1.67  ← learned words form sentences
Step  8000 | train: 1.43 | val: 1.62  ← writing like Shakespeare
```

From random noise to iambic pentameter in 8,000 steps. Wild.

---

## The things that surprised me

**Positional encoding** — I assumed transformers just "know" word order. They don't. Without positional encoding, "dog bit man" and "man bit dog" are literally identical to the model. I discovered this myself when I made two identical word vectors and watched their attention patterns become exactly the same.

**Causal masking** — The model wants to cheat. During training it can technically see future tokens. The triangular mask is what forces it to actually learn instead of just memorizing. One matrix of 1s and 0s is the difference between a language model and a lookup table.

**Residual connections** — The single line `X = X + attention(X)` is why transformers can be deep at all. Without it gradients vanish. With it, even a 12-layer network trains cleanly. Simple idea, massive impact.

**`-inf` before softmax** — I spent 20 minutes confused why we use `-inf` instead of just a big negative number like `-9999`. Then I realized: `e^(-9999)` is tiny but nonzero. `e^(-inf)` is exactly zero. One character of difference, completely different behavior.

You learn differently when you build it yourself.

---

## Run it

```bash
git clone https://github.com/manicc/shakespeare-gpt
cd shakespeare-gpt
pip install torch gradio
python app.py
```

Or skip setup → **[Live demo](https://huggingface.co/spaces/manicc/shakespeare-gpt)**

---

## Stack

`PyTorch` · `Gradio` · `HuggingFace Spaces` · `Google Colab T4`

---

## What's next

This was about understanding the internals. I now understand them.

Next project: building something actually useful with that knowledge. A RAG system — retrieval augmented generation — where I take these same transformer concepts and build something people would genuinely use.

Stay tuned.

---

*Built this because tutorials weren't enough.*
