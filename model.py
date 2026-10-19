import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 128
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(
            x,
            self.weight.shape,
            self.weight,
            self.bias,
            1e-5
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, c):
        super().__init__()

        assert c.n_embd % c.n_head == 0

        self.c_attn = nn.Linear(
            c.n_embd,
            3 * c.n_embd,
            bias=c.bias
        )

        self.c_proj = nn.Linear(
            c.n_embd,
            c.n_embd,
            bias=c.bias
        )

        self.attn_dropout = nn.Dropout(c.dropout)
        self.resid_dropout = nn.Dropout(c.dropout)

        self.n_head = c.n_head
        self.n_embd = c.n_embd

        self.flash = hasattr(
            F,
            "scaled_dot_product_attention"
        )

        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(
                    torch.ones(
                        c.block_size,
                        c.block_size
                    )
                ).view(
                    1,
                    1,
                    c.block_size,
                    c.block_size
                )
            )

    def forward(self, x):

        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(
            self.n_embd,
            dim=2
        )

        k = k.view(
            B,
            T,
            self.n_head,
            C // self.n_head
        ).transpose(1, 2)

        q = q.view(
            B,
            T,
            self.n_head,
            C // self.n_head
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_head,
            C // self.n_head
        ).transpose(1, 2)

        if self.flash:

            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=(
                    self.attn_dropout.p
                    if self.training
                    else 0.0
                ),
                is_causal=True
            )

        else:

            a = (
                q @ k.transpose(-2, -1)
            ) * (
                1.0 / math.sqrt(k.size(-1))
            )

            a = a.masked_fill(
                self.bias[:, :, :T, :T] == 0,
                float("-inf")
            )

            a = self.attn_dropout(
                F.softmax(a, dim=-1)
            )

            y = a @ v

        y = y.transpose(
            1,
            2
        ).contiguous().view(
            B,
            T,
            C
        )

        return self.resid_dropout(
            self.c_proj(y)
        )


class MLP(nn.Module):

    def __init__(self, c):
        super().__init__()

        self.c_fc = nn.Linear(
            c.n_embd,
            4 * c.n_embd,
            bias=c.bias
        )

        self.gelu = nn.GELU()

        self.c_proj = nn.Linear(
            4 * c.n_embd,
            c.n_embd,
            bias=c.bias
        )

        self.dropout = nn.Dropout(
            c.dropout
        )

    def forward(self, x):

        return self.dropout(
            self.c_proj(
                self.gelu(
                    self.c_fc(x)
                )
            )
        )


class Block(nn.Module):

    def __init__(self, c):
        super().__init__()

        self.ln1 = LayerNorm(
            c.n_embd,
            c.bias
        )

        self.attn = CausalSelfAttention(c)

        self.ln2 = LayerNorm(
            c.n_embd,
            c.bias
        )

        self.mlp = MLP(c)

    def forward(self, x):

        x = x + self.attn(
            self.ln1(x)
        )

        x = x + self.mlp(
            self.ln2(x)
        )

        return x


class GPT(nn.Module):

    def __init__(self, c):
        super().__init__()

        self.config = c

        self.transformer = nn.ModuleDict({

            "wte": nn.Embedding(
                c.vocab_size,
                c.n_embd
            ),

            "wpe": nn.Embedding(
                c.block_size,
                c.n_embd
            ),

            "drop": nn.Dropout(
                c.dropout
            ),

            "h": nn.ModuleList(
                [
                    Block(c)
                    for _ in range(c.n_layer)
                ]
            ),

            "ln_f": LayerNorm(
                c.n_embd,
                c.bias
            )
        })

        self.lm_head = nn.Linear(
            c.n_embd,
            c.vocab_size,
            bias=False
        )

        # Weight tying
        self.transformer["wte"].weight = (
            self.lm_head.weight
        )

    def forward(self, idx):

        _, t = idx.size()

        if t > self.config.block_size:

            idx = idx[
                :,
                -self.config.block_size:
            ]

            t = idx.size(1)

        pos = torch.arange(
            t,
            dtype=torch.long,
            device=idx.device
        )

        x = (
            self.transformer["wte"](idx)
            +
            self.transformer["wpe"](pos)
        )

        x = self.transformer["drop"](x)

        for block in self.transformer["h"]:
            x = block(x)

        x = self.transformer["ln_f"](x)

        return self.lm_head(
            x[:, [-1], :]
        )

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=0.8,
        top_k=40
    ):

        for _ in range(max_new_tokens):

            if idx.size(1) <= self.config.block_size:

                input_idx = idx

            else:

                input_idx = idx[
                    :,
                    -self.config.block_size:
                ]

            logits = self(
                input_idx
            )[:, -1, :]

            logits = logits / max(
                temperature,
                1e-5
            )

            if top_k:

                vals, _ = torch.topk(
                    logits,
                    min(
                        top_k,
                        logits.size(-1)
                    )
                )

                logits[
                    logits < vals[:, [-1]]
                ] = -float("Inf")

            probs = F.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat(
                (idx, next_token),
                dim=1
            )

        return idx