import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Patch Embedding
# -------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_size=128, img_size=64):
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

# -------------------------
# Transformer Expert
# -------------------------
class Expert(nn.Module):
    def __init__(self, emb_size, num_heads, hidden_size, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, emb_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + self.dropout(h)
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + self.dropout(h)
        return x

# -------------------------
# Router
# -------------------------
class Router(nn.Module):
    def __init__(self, emb_size, num_experts):
        super().__init__()
        self.gate = nn.Linear(emb_size, num_experts)

    def forward(self, x):
        # x: [B, N, E]
        # Returns softmax probabilities per token
        return F.softmax(self.gate(x), dim=-1)  # [B, N, num_experts]

# -------------------------
# MoE Layer (Parallel Experts)
# -------------------------
class MoE(nn.Module):
    def __init__(self, emb_size, num_heads, num_experts=4, hidden_size=None, dropout=0.1):
        super().__init__()
        hidden_size = hidden_size or emb_size * 4
        self.num_experts = num_experts
        self.router = Router(emb_size, num_experts)
        self.experts = nn.ModuleList([
            Expert(emb_size, num_heads, hidden_size, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: [B, N, E]
        B, N, E = x.shape

        # Get routing probabilities
        gates = self.router(x)  # [B, N, K]

        # Run all experts on all tokens
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x).unsqueeze(-2))  # [B, N, 1, E]

        # Stack experts: [B, N, K, E]
        expert_outputs = torch.cat(expert_outputs, dim=-2)

        # Weighted sum by gate probabilities
        gates = gates.unsqueeze(-1)          # [B, N, K, 1]
        out = (expert_outputs * gates).sum(dim=-2)  # [B, N, E]

        return out

# -------------------------
# ViT-MoE (Router -> Parallel Experts -> Average -> Classify)
# -------------------------
class ViTMoE(nn.Module):
    def __init__(
        self,
        img_size=64,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        emb_size=128,
        num_heads=4,
        hidden_ratio=4,
        num_experts=4,
        dropout=0.1
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, emb_size))
        self.dropout = nn.Dropout(dropout)

        # Single MoE layer
        self.moe = MoE(
            emb_size=emb_size,
            num_heads=num_heads,
            num_experts=num_experts,
            hidden_size=int(emb_size * hidden_ratio),
            dropout=dropout
        )

        self.norm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)                       # [B, N, E]
        cls = self.cls_token.expand(B, -1, -1)       # [B, 1, E]
        x = torch.cat([cls, x], dim=1)               # [B, N+1, E]
        x = x + self.pos_embed
        x = self.dropout(x)

        # MoE: Router -> Parallel Experts
        x = self.moe(x)

        x = self.norm(x)
        cls_out = x[:, 0]                             # [B, E]
        return self.head(cls_out)

# -------------------------
# Quick test
# -------------------------
if __name__ == "__main__":
    model = ViTMoE(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        emb_size=128,
        num_heads=4,
        hidden_ratio=2,
        num_experts=4,
        dropout=0.1
    )
    x = torch.randn(8, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)  # [8, 10]
