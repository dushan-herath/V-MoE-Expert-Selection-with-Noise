import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------
# Patch Embedding
# ----------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_size=128, img_size=32):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            emb_size,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)      # [B, E, H', W']
        x = x.flatten(2)      # [B, E, N]
        x = x.transpose(1, 2) # [B, N, E]
        return x


# ----------------------------------------------------
# Expert FFN
# ----------------------------------------------------
class Expert(nn.Module):
    def __init__(self, emb_size, hidden_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------
# Vectorized Top-K MoE (with optional routing output)
# ----------------------------------------------------
class MoE(nn.Module):
    def __init__(self, emb_size, num_experts=4, hidden_size=None, k=1, dropout=0.1):
        super().__init__()
        self.k = k
        self.num_experts = num_experts
        hidden_size = hidden_size or int(emb_size * 1.5)

        self.experts = nn.ModuleList([
            Expert(emb_size, hidden_size, dropout)
            for _ in range(num_experts)
        ])

        self.router = nn.Linear(emb_size, num_experts)

    def forward(self, x, return_routing=False):
        """
        x: [B, N, E]
        return_routing: if True, returns top-k expert indices for analysis
        """
        B, N, E = x.shape
        T = B * N  # total tokens

        # 1. Top-K routing (hard, sparse)
        logits = self.router(x)                     # [B, N, num_experts]
        _, topk_idx = torch.topk(logits, self.k, dim=-1)  # [B, N, k]

        flat_x = x.reshape(T, E)                      # [T, E]
        flat_topk = topk_idx.reshape(T, self.k)       # [T, k]

        # Output buffer
        out = torch.zeros_like(flat_x)             # [T, E]

        # 2. Selective expert execution
        for expert_id, expert in enumerate(self.experts):
            mask = (flat_topk == expert_id)         # [T, k]
            token_mask = mask.any(dim=1)            # [T]

            if not token_mask.any():
                continue

            tokens = flat_x[token_mask]             # [M, E]
            expert_out = expert(tokens)             # [M, E]

            # Count how many times token was routed here
            counts = mask[token_mask].sum(dim=1, keepdim=True)  # [M, 1]
            out[token_mask] += expert_out * counts

        # 3. Average over top-K experts
        out = out / self.k
        out = out.view(B, N, E)

        if return_routing:
            return out, topk_idx  # <-- new minimal change

        return out


# ----------------------------------------------------
# Self-Attention Block (with residual)
# ----------------------------------------------------
class SelfAttentionBlock(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(
            emb_size, num_heads, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(
            self.norm(x),
            self.norm(x),
            self.norm(x)
        )
        return x + self.dropout(attn_out)


# ----------------------------------------------------
# CLS-Free ViT-MoE
# ----------------------------------------------------
class ViTMoE(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        emb_size=128,
        num_heads=4,
        mlp_ratio=6.0,
        dropout=0.1,
        k=2
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            in_channels, patch_size, emb_size, img_size
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.n_patches, emb_size)
        )
        self.dropout = nn.Dropout(dropout)

        self.attn_block = SelfAttentionBlock(
            emb_size, num_heads, dropout
        )

        self.moe = MoE(
            emb_size=emb_size,
            num_experts=8,
            hidden_size=int(emb_size * mlp_ratio),
            dropout=dropout,
            k=k
        )

        self.norm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, return_routing=False):
        x = self.patch_embed(x)      # [B, N, E]
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.attn_block(x)

        if return_routing:
            x, routing = self.moe(x, return_routing=True)
        else:
            x = self.moe(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        logits = self.head(x)

        if return_routing:
            return logits, routing  # <-- minimal change
        return logits


# ----------------------------------------------------
# Quick Test
# ----------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViTMoE(k=2).to(device)

    x = torch.randn(8, 3, 32, 32).to(device)
    y = model(x, return_routing=True)
    print("Output shape:", y[0].shape)  # logits
    print("Routing shape:", y[1].shape) # top-k experts
