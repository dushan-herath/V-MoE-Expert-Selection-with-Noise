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
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, x):
        x = self.proj(x)          # [B, E, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, E]
        x = self.norm(x)
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
        # Xavier init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------
# Vectorized Top-k MoE
# ----------------------------------------------------
class MoE(nn.Module):
    def __init__(self, emb_size, num_experts=4, hidden_size=None, dropout=0.1, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        hidden_size = hidden_size or int(emb_size * 1.5)

        self.experts = nn.ModuleList([
            Expert(emb_size, hidden_size, dropout) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(emb_size, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [B, N, E]
        B, N, E = x.shape
        scores = self.gate(x)                   # [B, N, num_experts]
        topk_vals, topk_idx = torch.topk(scores, self.k, dim=-1)
        gates = self.softmax(topk_vals)         # [B, N, k]

        # Prepare expert outputs
        expert_inputs = [torch.zeros_like(x) for _ in range(self.num_experts)]
        for i in range(self.k):
            idx = topk_idx[..., i]              # [B, N]
            gate = gates[..., i].unsqueeze(-1)  # [B, N, 1]
            for e in range(self.num_experts):
                mask = (idx == e).unsqueeze(-1) # [B, N, 1]
                if mask.any():
                    out = self.experts[e](x[mask.squeeze(-1)])
                    expert_inputs[e][mask] += gate[mask] * out

        out = sum(expert_inputs)
        return out


# ----------------------------------------------------
# PreNorm Attention + MLP Block
# ----------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads, mlp_ratio=4.0, dropout=0.1, num_experts=4, k=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, num_heads, batch_first=True)
        self.drop_path = nn.Dropout(dropout)

        # Shared MLP
        hidden = int(emb_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, emb_size),
            nn.Dropout(dropout)
        )

        # MoE
        self.moe_norm = nn.LayerNorm(emb_size)
        self.moe = MoE(emb_size, num_experts=num_experts, hidden_size=hidden, dropout=dropout, k=k)

    def forward(self, x):
        # Attention
        x_res = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = x_res + self.drop_path(attn_out)

        # MLP
        x = x + self.mlp(x)

        # MoE
        x = x + self.moe(self.moe_norm(x))
        return x


# ----------------------------------------------------
# ViT-MoE
# ----------------------------------------------------
class ViTMoE(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 emb_size=128, depth=6, num_heads=4, mlp_ratio=4.0, dropout=0.1,
                 num_experts=4, k=2):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, emb_size))
        self.dropout = nn.Dropout(dropout)

        # Stack Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(emb_size, num_heads, mlp_ratio, dropout, num_experts, k)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, num_classes)

        # Init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)

        cls_out = self.norm(x)[:, 0]
        return self.head(cls_out)


# ----------------------------------------------------
# Quick test
# ----------------------------------------------------
if __name__ == "__main__":
    model = ViTMoE(img_size=32, patch_size=4, emb_size=128, depth=4, num_heads=4,
                   num_experts=2, k=1)
    x = torch.randn(8, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)
