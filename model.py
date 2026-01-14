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
        self.net = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------
# Vectorized Top-K MoE
# ----------------------------------------------------
class MoE(nn.Module):
    def __init__(self, emb_size, num_experts=4, hidden_size=None, k=1, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        hidden_size = hidden_size or int(emb_size * 1.5)

        # Experts
        self.experts = nn.ModuleList([
            Expert(emb_size, hidden_size, dropout)
            for _ in range(num_experts)
        ])

        # Router
        self.router = nn.Linear(emb_size, num_experts)

    def forward(self, x):
        # x: [B, N, E]
        B, N, E = x.shape

        # Compute gate scores
        scores = F.softmax(self.router(x), dim=-1)  # [B, N, num_experts]

        # Top-k indices and values
        topk_vals, topk_idx = torch.topk(scores, self.k, dim=-1)  # [B, N, k]

        # Flatten batch & sequence for indexing
        flat_x = x.reshape(B * N, E)                  # [B*N, E]
        flat_idx = topk_idx.reshape(B * N * self.k)  # [B*N*k]

        # Compute all expert outputs at once
        expert_outputs = torch.stack([expert(flat_x) for expert in self.experts], dim=1)  # [B*N, num_experts, E]

        # Gather top-k expert outputs
        batch_idx = torch.arange(B * N, device=x.device).repeat_interleave(self.k)          # [B*N*k]
        topk_outs = expert_outputs[batch_idx, flat_idx]                                     # [B*N*k, E]

        # Reshape and average top-k outputs
        topk_outs = topk_outs.view(B, N, self.k, E).mean(dim=2)                             # [B, N, E]

        return topk_outs, topk_idx


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
        x = self.norm(x)
        return x 


# ----------------------------------------------------
# ViT-MoE (1 Attention + Pre-MLP + Vectorized Top-k MoE)
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
        moe_layers=None,
        k=3
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
            num_experts=4,
            hidden_size=int(emb_size * mlp_ratio),
            dropout=dropout,
            k=k
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

        # 2️⃣ Shared MLP (optional)
        # x = x + self.pre_mlp(x)

        # 3️⃣ Top-k MoE
        x, topk_idx = self.moe(x)

        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)


# ----------------------------------------------------
# Quick test
# ----------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViTMoE(k=2).to(device)  # Top-2 experts
    x = torch.randn(8, 3, 32, 32).to(device)
    y = model(x)
    print("Output shape:", y.shape)
