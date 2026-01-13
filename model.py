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
        self.proj = nn.Conv2d(
            in_channels, emb_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)                  # [B, E, H', W']
        x = x.flatten(2)                  # [B, E, N]
        x = x.transpose(1, 2)             # [B, N, E]
        return x

# ----------------------------------------------------
# Expert Attention + FFN
# ----------------------------------------------------
class ExpertBlock(nn.Module):
    def __init__(self, emb_size, num_heads, hidden_size, dropout=0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_size, num_heads=num_heads, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Attention
        x_res = x
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x_res + attn_out

        # MLP
        x = x + self.mlp(x)
        return x

# ----------------------------------------------------
# Top-k MoE Layer
# ----------------------------------------------------
class MoE(nn.Module):
    def __init__(self, emb_size, num_experts=4, hidden_size=None, num_heads=4, k=1, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        hidden_size = hidden_size or int(emb_size * 1.5)

        # Each expert has its own attention + FFN
        self.experts = nn.ModuleList([
            ExpertBlock(emb_size, num_heads, hidden_size, dropout)
            for _ in range(num_experts)
        ])

        # Router
        self.router = nn.Linear(emb_size, num_experts)

    def forward(self, x):
        # x: [B, N, E]
        B, N, E = x.shape

        # Compute routing scores
        scores = F.softmax(self.router(x), dim=-1)  # [B, N, num_experts]
        topk_vals, topk_idx = torch.topk(scores, self.k, dim=-1)  # [B, N, k]

        # Initialize output tensor
        out = torch.zeros_like(x)

        # Apply top-k experts per token
        for b in range(B):
            for n in range(N):
                token = x[b, n:n+1]  # [1, E]
                experts_out = []
                for idx in topk_idx[b, n]:
                    experts_out.append(self.experts[idx](token))
                # Weighted average (equal weighting)
                out[b, n] = sum(experts_out) / self.k

        return out

# ----------------------------------------------------
# ViT-MoE with per-expert attention
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
        mlp_ratio=4.0,
        dropout=0.1,
        num_experts=4,
        top_k=1
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

        # MoE with per-expert attention + FFN
        self.moe = MoE(
            emb_size=emb_size,
            num_experts=num_experts,
            hidden_size=int(emb_size * mlp_ratio),
            num_heads=num_heads,
            k=top_k,
            dropout=dropout
        )

        self.norm = nn.LayerNorm(emb_size*2)
        self.head = nn.Linear(emb_size*2, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)

        x = self.patch_embed(x)                  # [B, N, E]
        cls = self.cls_token.expand(B, -1, -1)   # [B, 1, E]
        x = torch.cat([cls, x], dim=1)           # [B, N+1, E]
        x = x + self.pos_embed
        x = self.dropout(x)

        # Top-k MoE
        x = self.moe(x)

        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)

# ----------------------------------------------------
# Quick test
# ----------------------------------------------------
if __name__ == "__main__":
    model = ViTMoE(num_experts=4, top_k=2)
    x = torch.randn(8, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)
