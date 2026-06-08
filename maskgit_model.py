import torch
import torch.nn as nn
import torch.nn.functional as F
from configuration import ModelConfig
from utils import get_2d_sinusoidal_pos_embed


class MLP(nn.Module):
    """Two-layer feed-forward network: Linear -> GELU -> Linear."""

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1  = nn.Linear(embed_dim, hidden_dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2  = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, embed_dim)
        x = self.fc1(x)    # (B, N, hidden_dim)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)    # (B, N, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head bi-directional attention."""

    def __init__(self, num_heads: int, input_dim: int, attn_drop: float = 0.0):
        super().__init__()
        assert input_dim % num_heads == 0
        self.num_h    = num_heads
        self.head_dim = input_dim // num_heads
        self.scale    = self.head_dim ** -0.5

        self.k         = nn.Linear(input_dim, input_dim, bias=False)
        self.q         = nn.Linear(input_dim, input_dim, bias=False)
        self.v         = nn.Linear(input_dim, input_dim, bias=False)
        self.out       = nn.Linear(input_dim, input_dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, _ = inputs.shape

        k = self.k(inputs).reshape(B, N, self.num_h, self.head_dim).transpose(1, 2)
        v = self.v(inputs).reshape(B, N, self.num_h, self.head_dim).transpose(1, 2)
        q = self.q(inputs).reshape(B, N, self.num_h, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, N, N)

        # mask: (B, N) True = this key position is masked (excluded from attention)
        # expand to (B, 1, 1, N) to broadcast over heads and query positions
        if mask is not None:
            attn = attn.masked_fill(mask[:, None, None, :], float("-inf"))

        attn = self.attn_drop(F.softmax(attn, dim=-1))  # (B, heads, N, N)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)  # (B, N, D)
        return self.out(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN + Attention + LN + MLP."""

    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadAttention(num_heads, embed_dim, attn_drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = MLP(embed_dim, mlp_ratio, proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x

class MixedSelfAttention(nn.Module):
    """Self-attention where the first n_global tokens use their own Q/K/V projections."""

    def __init__(self, embed_dim: int, num_heads: int, n_global: int = 2,
                 attn_drop: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_h    = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale    = self.head_dim ** -0.5
        self.n_global = n_global

        # Separate Q/K/V for global tokens
        self.q_global = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_global = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_global = nn.Linear(embed_dim, embed_dim, bias=False)
        # Separate Q/K/V for patch tokens
        self.q_patch  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_patch  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_patch  = nn.Linear(embed_dim, embed_dim, bias=False)

        self.out       = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, _ = inputs.shape
        ng      = self.n_global
        g, p    = inputs[:, :ng, :], inputs[:, ng:, :]

        # Project Q/K/V for each group, then concatenate
        q = torch.cat([self.q_global(g), self.q_patch(p)], dim=1)  # (B, N, D)
        k = torch.cat([self.k_global(g), self.k_patch(p)], dim=1)
        v = torch.cat([self.v_global(g), self.v_patch(p)], dim=1)

        q = q.reshape(B, N, self.num_h, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_h, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_h, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale              # (B, heads, N, N)
        if mask is not None:
            attn = attn.masked_fill(mask[:, None, None, :], float("-inf"))
        attn = self.attn_drop(F.softmax(attn, dim=-1))

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.out(x)


class MixedTransformerBlock(nn.Module):
    """Pre-norm transformer block using MixedSelfAttention."""

    def __init__(self, embed_dim: int, num_heads: int, n_global: int = 2,
                 mlp_ratio: float = 4.0, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MixedSelfAttention(embed_dim, num_heads, n_global, attn_drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = MLP(embed_dim, mlp_ratio, proj_drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class PredictNext(nn.Module):
    """Predict the indexes of the next batch of tokens to be generated.

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.l_tokens = nn.Linear(input_dim,input_dim)
        self.l_head = nn.Linear(2 * input_dim, input_dim)
        
    # potential input, class_embed: torch.Tensor,
    def forward(self,  inputs: torch.Tensor):
        """
        Args:
            inputs (torch.Tensor): (B, N, D)

        Returns:
            output (torch.Tensor): (B, N, 1)
        """
        B = inputs.shape[0]
        global_input = inputs[:, 0:2, :].view(B, 1, -1)
        
        h = self.l_head(global_input).permute(0, 2, 1)
        tokens = self.l_tokens(inputs[:, 2:, :])
        return tokens@h
    

class MaskedViT(nn.Module):
    """Bidirectional masked transformer for MaskGIT-style token prediction.

    Context embedding = sinusoidal 2D position embedding
                      + CLIP image class embedding (projected, broadcast over all tokens)
    """

    def __init__(self, cfg: ModelConfig = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg

        grid_size     = cfg.img_size // 16
        codebook_size = cfg.codebook_size

        # token embedding (+1 index reserved for [MASK] token)
        self.embed = nn.Embedding(codebook_size + 1, cfg.embed_dim)

        # sinusoidal 2D positional embedding -- fixed, non-trainable
        self.register_buffer(
            "pos_embed",
            get_2d_sinusoidal_pos_embed(cfg.embed_dim, grid_size),  # (1, N, embed_dim)
        )

        # project CLIP image embedding -> embed_dim
        self.clip_proj = nn.Linear(cfg.clip_dim, cfg.embed_dim)

        # learnable step embedding (always-visible global token)
        self.step_embed = nn.Embedding(cfg.max_steps + 1, cfg.embed_dim)

        self.blocks = nn.ModuleList([
            MixedTransformerBlock(cfg.embed_dim, cfg.num_heads, n_global=2,
                                  mlp_ratio=cfg.mlp_ratio, attn_drop=cfg.attn_drop,
                                  proj_drop=cfg.proj_drop)
            for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.head = nn.Linear(cfg.embed_dim, codebook_size, bias=True)
        self.pred_next = PredictNext(cfg.embed_dim)

    def forward(self, tokens: torch.Tensor, query_mask: torch.Tensor, key_mask: torch.Tensor, clip_feat: torch.Tensor, step=None) -> torch.Tensor:
        """
        Args:
            tokens:    (B, N)         integer token ids
            clip_feat: (B, clip_dim)  CLIP class embedding
            query_mask:(B, N)
            key_mask:  (B, N)
            step:      (B,) int tensor or int — current unmasking step index

        Returns:
            logits:      (B, N, codebook_size)
            pred_next:   (B, N, 1)
        """
        B = tokens.shape[0]

        # normalise step to (B,) tensor
        if step is None:
            step = torch.zeros(B, dtype=torch.long, device=tokens.device)
        elif isinstance(step, int):
            step = torch.full((B,), step, dtype=torch.long, device=tokens.device)

        # --- token + positional embedding ------------------------------------
        x = self.embed(tokens) + self.pos_embed   # (B, N, D)

        # --- prepend two always-visible global tokens ------------------------
        # token 0: CLIP class embedding   token 1: step embedding
        clip_tok = self.clip_proj(clip_feat).unsqueeze(1)  # (B, 1, D)
        step_tok = self.step_embed(step).unsqueeze(1)      # (B, 1, D)
        x = torch.cat([clip_tok, step_tok, x], dim=1)     # (B, N+2, D)

        # --- attention mask: both global tokens always visible ---------------
        if self.cfg.disable_mask_in_attention:
            mask = None
        else:
            global_visible = torch.zeros(B, 2, dtype=torch.bool, device=x.device)
            mask = torch.cat([global_visible, ~(query_mask | key_mask)], dim=1)  # (B, N+2)

        # --- transformer blocks ---------------------------------------------
        for block in self.blocks:
            x = block(x, mask)                    # (B, N+2, D)

        x = self.norm(x)
        # step token (index 1) acts as the query head for pred_next
        predict_next = self.pred_next(x)   # (step + patches) → (B, N, 1)
        patch_x = x[:, 2:, :]                         # (B, N, D)  drop both global tokens
        return self.head(patch_x), predict_next


