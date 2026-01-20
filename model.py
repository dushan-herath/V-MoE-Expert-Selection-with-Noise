import torch
import torch.nn as nn
import torch.nn.functional as F


# Convert image into a sequence of patch embeddings
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_size=128, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.img_size = img_size

        # CNN stem for richer features (no downsampling)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, emb_size // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(emb_size // 2),
            nn.GELU(),

            nn.Conv2d(emb_size // 2, emb_size // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(emb_size // 2),
            nn.GELU(),

            # residual-like connection
            nn.Conv2d(emb_size // 2, emb_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(emb_size),
            nn.GELU()
        )

        # Patch projection (non-overlapping patches)
        self.proj = nn.Conv2d(
            emb_size,
            emb_size,
            kernel_size=patch_size,
            stride=patch_size
        )

        # compute number of patches
        self.n_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        # CNN stem extracts rich features
        x = self.stem(x)      # [B, E, 32, 32]
        x = self.proj(x)      # [B, E, 8, 8]
        B, E, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, 64, E]
        return x


# Expert feed forward network
class Expert(nn.Module):
    def __init__(self, emb_size, hidden_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, hidden_size), # sequential layers
            nn.GELU(),
            nn.Linear(hidden_size, emb_size), # converting back to emb_size
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# top k routing MoE layer
# This layer routes tokens to top-k experts based on router probabilities
# then executes only selected experts and combines outputs weighted by routing probs
class MoE(nn.Module):
    def __init__(self, emb_size, num_experts=4, hidden_size=None, k=1, dropout=0.1):
        super().__init__()
        self.k = k # select top k experts for each token
        self.num_experts = num_experts # total number of experts
        hidden_size = hidden_size or int(emb_size * 1.5)

        # list of expert networks
        self.experts = nn.ModuleList([
            Expert(emb_size, hidden_size, dropout)
            for _ in range(num_experts)
        ])

        # routing network outputting logits for each expert per token
        self.router = nn.Linear(emb_size, num_experts)

    def forward(self, x, return_routing=False, return_load_loss=False):
        B, N, E = x.shape
        T = B * N

        # routing logits for each token
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)    # convert logits to probabilities [B,N,num_experts]

        # top-k selection from probabilities
        topk_probs, topk_idx = torch.topk(probs, self.k, dim=-1)  # [B,N,k]

        # flatten tokens for processing
        flat_x = x.reshape(T, E)        # [T, E]
        flat_idx = topk_idx.reshape(T, self.k)  # [T, k]
        flat_w = topk_probs.reshape(T, self.k)  # [T, k]

        # initialize output
        out = torch.zeros_like(flat_x)

        # process each expert separately
        for expert_id, expert in enumerate(self.experts):
            mask = (flat_idx == expert_id)           # [T, k], tokens assigned to this expert
            token_mask = mask.any(dim=1)             # [T], True if token assigned to expert
            if not token_mask.any():
                continue

            tokens = flat_x[token_mask]              # select tokens for this expert
            expert_out = expert(tokens)              # process tokens through expert

            # routing weights for this expert
            weights = flat_w[token_mask][mask[token_mask]]  # [M], top-k weights for expert
            out[token_mask] += expert_out * weights.unsqueeze(1)  # weighted sum

        # reshape back to [B, N, E]
        out = out.view(B, N, E)

        # compute load balancing loss if requested
        if return_load_loss:
            # Importance: sum of probabilities per expert
            importance = probs.sum(dim=(0,1))  # [num_experts]
            importance = importance / importance.sum()

            # Load: count of tokens assigned to each expert in top-k
            expert_range = torch.arange(self.num_experts, device=x.device)  # [num_experts]
            load_mask = (flat_idx[:, None, :] == expert_range[None,:,None])  # [T,num_experts,k]
            load = load_mask.any(dim=2).sum(dim=0)                             # [num_experts]
            load = load / load.sum()

            load_loss = (importance * load * self.num_experts).sum()  # encourages balanced usage
            return out, topk_idx, load_loss

        if return_routing:
            return out, topk_idx

        return out


# Attention block
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
            self.norm(x),  # query
            self.norm(x),  # key
            self.norm(x)   # value
        )
        return x + self.dropout(attn_out)  # residual connection


# ViT-MoE model
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

        # patch embedding layer
        self.patch_embed = PatchEmbedding(
            in_channels, patch_size, emb_size, img_size
        )

        # positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.n_patches, emb_size)
        )
        self.dropout = nn.Dropout(dropout)

        # attention block
        self.attn_block = SelfAttentionBlock(
            emb_size, num_heads, dropout
        )

        # MoE layer
        self.moe = MoE(
            emb_size=emb_size,
            num_experts=6,
            hidden_size=int(emb_size * mlp_ratio),
            dropout=dropout,
            k=k
        )

        # normalization and classification head
        self.norm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, num_classes)

        # initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, return_routing=False, return_load_loss=False):
        # patch embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.dropout(x)

        # attention block
        x = self.attn_block(x)

        # MoE layer
        if return_routing or return_load_loss:
            moe_out = self.moe(x, return_routing=True, return_load_loss=return_load_loss)

            # unpack according to what is returned by MoE
            if return_load_loss:
                x, routing, load_loss = moe_out
            else:
                x, routing = moe_out
        else:
            x = self.moe(x)


        # normalization and pooling
        x = self.norm(x)
        x = x.mean(dim=1)
        logits = self.head(x)

        # return outputs according to flags
        if return_load_loss:
            return logits, routing, load_loss
        if return_routing:
            return logits, routing
        return logits


# test the model
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViTMoE(k=2).to(device)

    x = torch.randn(8, 3, 32, 32).to(device)
    y = model(x, return_routing=True)
    print("Output shape:", y[0].shape)  # logits
    print("Routing shape:", y[1].shape) # top k experts
