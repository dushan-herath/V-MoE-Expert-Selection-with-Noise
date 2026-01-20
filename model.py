import torch
import torch.nn as nn
import torch.nn.functional as F


# Convert image into a sequence of patch embeddings
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_size=128, img_size=32):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2

        # CNN stem for better low-level feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, emb_size // 2, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(emb_size // 2),
            nn.GELU(),
            nn.Conv2d(emb_size // 2, emb_size, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(emb_size),
            nn.GELU()
        )

        self.proj = nn.Conv2d(
            emb_size,               # input channels
            emb_size,               # output channels
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.stem(x)      # extract local features before patchification
        x = self.proj(x)      # [B,3,32,32] converts to [B, E, H', W'] ~ [B, emb_size(128), 8,8]
        x = x.flatten(2)      # flat image patches to [B, E, N] ~ [B, emb_size(128), 64]
        x = x.transpose(1, 2) # [B, N, E]
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
# First this layer routes tokens to experts based on top k probabilities
#   then it executes only those experts for the selected tokens
#   and combines the outputs weighted by the routing probabilities

class MoE(nn.Module):
    def __init__(self, emb_size, num_experts=4, hidden_size=None, k=1, dropout=0.1):
        super().__init__()
        self.k = k # select top k experts for each token
        self.num_experts = num_experts # total number of experts
        hidden_size = hidden_size or int(emb_size * 1.5) 

        self.experts = nn.ModuleList([  
            Expert(emb_size, hidden_size, dropout)  # creating list of experts
            for _ in range(num_experts)
        ])

        self.router = nn.Linear(emb_size, num_experts) # routing network outputting logits for each expert per token

    def forward(self, x, return_routing=False):

        B, N, E = x.shape # batch size, number of tokens, embedding size
        
        T = B * N # get total tokens in the batch

        logits = self.router(x)      # get routing logits for each token  [B, N, num_experts]

        if self.training: # added this to reduce expert collapse during training
            logits = logits + 0.1 * torch.randn_like(logits)  

        probs = F.softmax(logits, dim=-1)    # convert logits to probabilities [B, N, num_experts]

        # top k selection from probabilities
        topk_probs, topk_idx = torch.topk(probs, self.k, dim=-1)  # output is [B,N,k] for both probs and indices

        flat_x = x.reshape(T, E) # flat all tokens in the batch to [T, E]
        flat_idx = topk_idx.reshape(T, self.k) # flat all top k expert indices in the batch to [T, k]
        flat_w = topk_probs.reshape(T, self.k) # flat all top k expert weights in the batch to [T, k]

        out = torch.zeros_like(flat_x) # create a tensor to hold the output [T, E]

        # find for each expert, what tokens are assigned to it and process them and put them back
        for expert_id, expert in enumerate(self.experts):
            mask = (flat_idx == expert_id)        # [T, k] create mask for tokens assigned to this expert like [[F,F],[T,F] ......
            token_mask = mask.any(dim=1)   # [T] , if any row of the k selections is this expert, mark True, else False like [F,T,F,T,.......

            if not token_mask.any():    # if no tokens for this expert continue
                continue

            tokens = flat_x[token_mask]  # outputs [M, E] select tokens for this expert, M if True count in token_mask
            expert_out = expert(tokens)  # outputs [M, E] process tokens through each expert to get expert outputs which are emb_size

            # routing weights for this expert
            weights = flat_w[token_mask][mask[token_mask]]  # output size is  [M], select the corresponding weights for the selected tokens

            out[token_mask] += expert_out * weights.unsqueeze(1)  # weight the expert outputs by the routing weights and update the output

        out = out.view(B, N, E) # reshape total tokens output back to [B, N, E]

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
        attn_out, _ = self.attn( # query, key, value all are x after normalization
            self.norm(x), 
            self.norm(x),
            self.norm(x)
        )
        return x + self.dropout(attn_out) # residual connection


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
            in_channels, patch_size, emb_size, img_size  # patch embedding layer
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.n_patches, emb_size) # positional embeddings layer with shape [1, N, E]
        )
        self.dropout = nn.Dropout(dropout)

        self.attn_block = SelfAttentionBlock(
            emb_size, num_heads, dropout # attention block
        )

        self.moe = MoE(
            emb_size=emb_size,
            num_experts=6,
            hidden_size=int(emb_size * mlp_ratio), # moe layer
            dropout=dropout,
            k=k
        )

        self.norm = nn.LayerNorm(emb_size) # normalization layer
        self.head = nn.Linear(emb_size, num_classes) # classification head

        nn.init.trunc_normal_(self.pos_embed, std=0.02) # initial positional embeddings

    def forward(self, x, return_routing=False):
        x = self.patch_embed(x)       # generate patch embeddings
        x = x + self.pos_embed  # add positional embeddings
        x = self.dropout(x)      # apply dropout

        x = self.attn_block(x)    # apply attention block

        if return_routing:  # if routing info is requested
            x, routing = self.moe(x, return_routing=True)
        else:
            x = self.moe(x)     # apply MoE layer output still [B, N, E]

        x = self.norm(x)      # normalize output 
        x = x.mean(dim=1)   # global average pooling over tokens -> [B, E]
        logits = self.head(x)   # classification head -> [B, num_classes]

        if return_routing:
            return logits, routing # return logits and routing info
        return logits


# test the model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViTMoE(k=2).to(device)

    x = torch.randn(8, 3, 32, 32).to(device)
    y = model(x, return_routing=True)
    print("Output shape:", y[0].shape)  # logits
    print("Routing shape:", y[1].shape) # top k experts
