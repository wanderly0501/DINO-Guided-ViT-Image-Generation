"""
Tests for maskgit_model.py.
Run with:  python maskgit_model_test.py
"""

import torch
from configuration import ModelConfig
from maskgit_model import MLP, MultiHeadAttention, TransformerBlock, PredictNext, MaskedViT

# ── small config so tests run fast ──────────────────────────────────────────
B          = 2      # batch size
N          = 16     # token sequence length
EMBED_DIM  = 64
NUM_HEADS  = 4
CLIP_DIM   = 32
CODEBOOK   = 8      # small codebook for testing


def small_cfg() -> ModelConfig:
    cfg = ModelConfig()
    cfg.embed_dim     = EMBED_DIM
    cfg.num_heads     = NUM_HEADS
    cfg.depth         = 2
    cfg.clip_dim      = CLIP_DIM
    cfg.codebook_size = CODEBOOK
    cfg.img_size      = 64    # 64 // 16 = 4 -> grid 4x4 = 16 tokens
    cfg.mlp_ratio     = 2.0
    cfg.attn_drop     = 0.0
    cfg.proj_drop     = 0.0
    return cfg


# ── helpers ──────────────────────────────────────────────────────────────────

def rand_mask(b, n, masked_ratio=0.5) -> torch.Tensor:
    """Boolean mask (B, N): True = masked position."""
    return torch.rand(b, n) < masked_ratio


# ── MLP ──────────────────────────────────────────────────────────────────────

def test_mlp_output_shape():
    mlp = MLP(embed_dim=EMBED_DIM, mlp_ratio=2.0)
    x   = torch.randn(B, N, EMBED_DIM)
    out = mlp(x)
    assert out.shape == (B, N, EMBED_DIM), f"MLP output shape wrong: {out.shape}"
    print(f"[PASS] MLP output: {tuple(out.shape)}")


def test_mlp_residual_invariant():
    """Output dtype and device match input."""
    mlp = MLP(embed_dim=EMBED_DIM)
    x   = torch.randn(B, N, EMBED_DIM)
    out = mlp(x)
    assert out.dtype  == x.dtype
    assert out.device == x.device
    print("[PASS] MLP dtype/device preserved")


# ── MultiHeadAttention ───────────────────────────────────────────────────────

def test_mha_output_shape_no_mask():
    mha = MultiHeadAttention(num_heads=NUM_HEADS, input_dim=EMBED_DIM)
    x   = torch.randn(B, N, EMBED_DIM)
    out = mha(x, mask=None)
    assert out.shape == (B, N, EMBED_DIM), f"MHA (no mask) shape wrong: {out.shape}"
    print(f"[PASS] MHA no-mask output: {tuple(out.shape)}")


def test_mha_output_shape_with_mask():
    mha  = MultiHeadAttention(num_heads=NUM_HEADS, input_dim=EMBED_DIM)
    x    = torch.randn(B, N, EMBED_DIM)
    mask = rand_mask(B, N)               # (B, N) bool
    out  = mha(x, mask=mask)
    assert out.shape == (B, N, EMBED_DIM), f"MHA (with mask) shape wrong: {out.shape}"
    print(f"[PASS] MHA with-mask output: {tuple(out.shape)}")


def test_mha_mask_affects_output():
    """Masking a key position should change the output."""
    mha  = MultiHeadAttention(num_heads=NUM_HEADS, input_dim=EMBED_DIM)
    x    = torch.randn(B, N, EMBED_DIM)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, 0] = True                    # mask first token as key

    out_masked   = mha(x, mask=mask)
    out_unmasked = mha(x, mask=None)
    assert not torch.allclose(out_masked, out_unmasked), \
        "Mask had no effect on attention output"
    print("[PASS] MHA mask changes output")


# ── TransformerBlock ─────────────────────────────────────────────────────────

def test_transformer_block_output_shape():
    block = TransformerBlock(embed_dim=EMBED_DIM, num_heads=NUM_HEADS)
    x     = torch.randn(B, N, EMBED_DIM)
    mask  = rand_mask(B, N)
    out   = block(x, mask)
    assert out.shape == (B, N, EMBED_DIM), f"TransformerBlock shape wrong: {out.shape}"
    print(f"[PASS] TransformerBlock output: {tuple(out.shape)}")


# ── PredictNext ──────────────────────────────────────────────────────────────

def test_predict_next_output_shape():
    pred = PredictNext(input_dim=EMBED_DIM)
    x    = torch.randn(B, N + 2, EMBED_DIM)
    out  = pred(x)
    assert out.shape == (B, N, 1), f"PredictNext shape wrong: {out.shape}"
    print(f"[PASS] PredictNext output: {tuple(out.shape)}")


# ── MaskedViT (full model) ───────────────────────────────────────────────────

def test_maskedvit_output_shapes():
    cfg        = small_cfg()
    model      = MaskedViT(cfg)
    model.eval()

    tokens     = torch.randint(0, CODEBOOK, (B, N))          # (B, N)
    query_mask = rand_mask(B, N, masked_ratio=0.5)            # (B, N) bool
    key_mask   = rand_mask(B, N, masked_ratio=0.3)            # (B, N) bool
    clip_feat  = torch.randn(B, CLIP_DIM)                     # (B, clip_dim)

    with torch.no_grad():
        logits, pred_next = model(tokens, query_mask, key_mask, clip_feat, step=0)

    assert logits.shape    == (B, N, CODEBOOK), f"logits shape wrong: {logits.shape}"
    assert pred_next.shape == (B, N, 1),        f"pred_next shape wrong: {pred_next.shape}"
    print(f"[PASS] MaskedViT logits:    {tuple(logits.shape)}")
    print(f"[PASS] MaskedViT pred_next: {tuple(pred_next.shape)}")


def test_maskedvit_mask_token():
    """[MASK] token id = codebook_size should be a valid embedding index."""
    cfg   = small_cfg()
    model = MaskedViT(cfg)
    model.eval()

    # all tokens masked
    tokens     = torch.full((B, N), fill_value=CODEBOOK, dtype=torch.long)
    query_mask = torch.ones(B, N, dtype=torch.bool)
    key_mask   = torch.ones(B, N, dtype=torch.bool)
    clip_feat  = torch.randn(B, CLIP_DIM)

    with torch.no_grad():
        logits, _ = model(tokens, query_mask, key_mask, clip_feat, step=0)

    assert logits.shape == (B, N, CODEBOOK)
    print("[PASS] MaskedViT handles all-[MASK] input")


def test_maskedvit_global_token_influences_output():
    """Different clip_feat must produce different logits (global token reaches attention)."""
    cfg   = small_cfg()
    model = MaskedViT(cfg)
    model.eval()

    tokens     = torch.randint(0, CODEBOOK, (B, N))
    query_mask = rand_mask(B, N, masked_ratio=0.5)
    key_mask   = torch.zeros(B, N, dtype=torch.bool)
    clip_feat1 = torch.randn(B, CLIP_DIM)
    clip_feat2 = torch.randn(B, CLIP_DIM)

    with torch.no_grad():
        logits1, _ = model(tokens, query_mask, key_mask, clip_feat1, step=0)
        logits2, _ = model(tokens, query_mask, key_mask, clip_feat2, step=0)

    assert not torch.allclose(logits1, logits2), \
        "clip_feat change had no effect on logits — global token not influencing attention"
    print("[PASS] MaskedViT global token influences output")


def test_maskedvit_global_token_prevents_nan():
    """When all patch positions are excluded from attention, the global token prevents NaN."""
    cfg   = small_cfg()
    model = MaskedViT(cfg)
    model.eval()

    tokens     = torch.randint(0, CODEBOOK, (B, N))
    # query=False, key=False -> ~(F|F)=True for all patches -> all patches excluded
    query_mask = torch.zeros(B, N, dtype=torch.bool)
    key_mask   = torch.zeros(B, N, dtype=torch.bool)
    clip_feat  = torch.randn(B, CLIP_DIM)

    with torch.no_grad():
        logits, pred_next = model(tokens, query_mask, key_mask, clip_feat, step=0)

    assert torch.isfinite(logits).all(),    "logits contain NaN when all patches excluded"
    assert torch.isfinite(pred_next).all(), "pred_next contains NaN when all patches excluded"
    print("[PASS] MaskedViT global token prevents NaN when all patches excluded")


# ── run all ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_mlp_output_shape()
    test_mlp_residual_invariant()
    test_mha_output_shape_no_mask()
    test_mha_output_shape_with_mask()
    test_mha_mask_affects_output()
    test_transformer_block_output_shape()
    test_predict_next_output_shape()
    test_maskedvit_output_shapes()
    test_maskedvit_mask_token()
    test_maskedvit_global_token_influences_output()
    test_maskedvit_global_token_prevents_nan()
    print("\nAll tests passed.")
