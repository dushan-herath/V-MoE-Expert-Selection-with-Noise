import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------
# Patch embedding
# ----------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_size=128, img_size=32):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_size,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                  # [B, E, H', W']
        x = x.flatten(2)                  # [B, E, N]
        x = x.transpose(1, 2)             # [B, N, E]
        return x


# ----------------------------------------------------
# Expert FFN
# ----------------------------------------------------
class Expert(nn.Module):
    def __init__(self, emb_size, hidden_size, dropout):
        super().__init__()
        hidden_size2 = hidden_size * 2  # second hidden layer wider
        self.net = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size2),
            nn.GELU(),
            nn.Linear(hidden_size2, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------
# Top-k MoE
# ----------------------------------------------------
class MoE(nn.Module):
    def __init__(self, emb_size, num_experts=4, hidden_size=None, dropout=0.1, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        hidden_size = hidden_size or emb_size * 4

        self.experts = nn.ModuleList([
            Expert(emb_size, hidden_size, dropout)
            for _ in range(num_experts)
        ])

        self.gate = nn.Linear(emb_size, num_experts)

    def forward(self, x):
        # x: [B, N, E]
        B, N, E = x.shape

        scores = self.gate(x)                        # [B, N, K]
        topk_vals, topk_idx = torch.topk(scores, self.k, dim=-1)
        gates = F.softmax(topk_vals, dim=-1)        # [B, N, k]

        out = torch.zeros_like(x)

        for i in range(self.k):
            idx = topk_idx[..., i]                  # [B, N]
            gate = gates[..., i].unsqueeze(-1)      # [B, N, 1]

            for e in range(self.num_experts):
                mask = (idx == e)
                if mask.any():
                    expert_out = self.experts[e](x[mask])
                    out[mask] += gate[mask] * expert_out

        return out


# ----------------------------------------------------
# Single Attention Block
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
        x_res = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        x = self.dropout(x)
        return x + x_res


# ----------------------------------------------------
# ViT-MoE (1 Attention + Pre-MLP + Top-k MoE)
# ----------------------------------------------------
class ViTMoE(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        emb_size=128,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
        moe_layers=None
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            in_channels, patch_size, emb_size, img_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.n_patches + 1, emb_size)
        )
        self.dropout = nn.Dropout(dropout)

        self.attn_block = SelfAttentionBlock(
            emb_size=emb_size,
            num_heads=num_heads,
            dropout=dropout
        )

        # 🔹 Shared Pre-MLP
        self.pre_mlp = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, int(emb_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(emb_size * mlp_ratio), emb_size),
            nn.Dropout(dropout)
        )

        # 🔹 Top-k MoE
        self.moe = MoE(
            emb_size=emb_size,
            num_experts=10,
            hidden_size=int(emb_size * mlp_ratio),
            dropout=dropout,
            k=8
        )

        self.norm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)

        x = self.patch_embed(x)                  # [B, N, E]
        cls = self.cls_token.expand(B, -1, -1)   # [B, 1, E]
        x = torch.cat([cls, x], dim=1)           # [B, N+1, E]
        x = x + self.pos_embed
        x = self.dropout(x)

        # 1️⃣ Attention
        x = self.attn_block(x)

        # 2️⃣ Shared MLP
        x = x + self.pre_mlp(x)

        # 3️⃣ Top-k MoE
        x = self.moe(x)

        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)


# ----------------------------------------------------
# Quick test
# ----------------------------------------------------
if __name__ == "__main__":
    model = ViTMoE()
    x = torch.randn(8, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)
