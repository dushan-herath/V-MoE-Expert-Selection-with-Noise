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
            in_channels, emb_size,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)                  # [B, E, H', W']
        x = x.flatten(2).transpose(1, 2) # [B, N, E]
        return x


# ----------------------------------------------------
# Router (Top-K + z-loss)
# ----------------------------------------------------
class Router(nn.Module):
    def __init__(self, emb_size, num_experts, z_loss_coeff=1e-3):
        super().__init__()
        self.linear = nn.Linear(emb_size, num_experts)
        self.z_loss_coeff = z_loss_coeff

    def forward(self, x):
        logits = self.linear(x)
        probs = F.softmax(logits, dim=-1)
        z_loss = self.z_loss_coeff * (logits ** 2).mean()
        return probs, z_loss


# ----------------------------------------------------
# Mixture of Experts (Switch-style)
# ----------------------------------------------------
class MoE(nn.Module):
    def __init__(
        self,
        emb_size,
        num_experts=4,
        hidden_size=512,
        dropout=0.1,
        k=2,
        capacity_factor=1.25
    ):
        super().__init__()

        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor

        self.router = Router(emb_size, num_experts)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, emb_size)
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        B, T, D = x.shape
        probs, z_loss = self.router(x)

        topk_probs, topk_idx = torch.topk(probs, self.k, dim=-1)
        capacity = int(self.capacity_factor * B * T / self.num_experts)

        output = torch.zeros_like(x)
        expert_usage = torch.zeros(self.num_experts, device=x.device)

        for e in range(self.num_experts):
            mask = (topk_idx == e).any(dim=-1)
            tokens = x[mask]

            if tokens.shape[0] > capacity:
                tokens = tokens[:capacity]

            if tokens.shape[0] > 0:
                out = self.experts[e](tokens)
                output[mask][:out.shape[0]] += out
                expert_usage[e] += tokens.shape[0]

        load = expert_usage / (expert_usage.sum() + 1e-6)
        balance_loss = self.num_experts * (load * load).sum()

        return output, balance_loss + z_loss, expert_usage


# ----------------------------------------------------
# Transformer Block
# ----------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(
        self,
        emb_size,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.1,
        use_moe=False,
        num_experts=4,
        k=2
    ):
        super().__init__()

        self.use_moe = use_moe

        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(
            emb_size, num_heads, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(emb_size)

        if use_moe:
            self.ff = MoE(
                emb_size,
                num_experts,
                int(emb_size * mlp_ratio),
                dropout,
                k
            )
        else:
            self.ff = nn.Sequential(
                nn.Linear(emb_size, int(emb_size * mlp_ratio)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(emb_size * mlp_ratio), emb_size)
            )

        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(
            self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        )

        if self.use_moe:
            out, aux_loss, usage = self.ff(self.norm2(x))
            x = x + self.dropout2(out)
            return x, aux_loss, usage
        else:
            x = x + self.dropout2(self.ff(self.norm2(x)))
            return x, 0.0, None


# ----------------------------------------------------
# Vision Transformer with MoE
# ----------------------------------------------------
class ViT_MoE(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        emb_size=128,
        depth=6,
        num_heads=4,
        num_classes=10,
        num_experts=4,
        dropout=0.1
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            3, patch_size, emb_size, img_size
        )

        num_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, emb_size)
        )

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                emb_size,
                num_heads,
                use_moe=(i >= depth // 2),
                num_experts=num_experts
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        B = x.size(0)

        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        total_aux_loss = 0.0

        for block in self.blocks:
            x, aux_loss, _ = block(x)
            total_aux_loss += aux_loss

        x = self.norm(x)
        cls_out = x[:, 0]
        logits = self.head(cls_out)

        return logits, total_aux_loss
