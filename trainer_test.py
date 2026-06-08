"""
Sanity-check tests for trainer.py.
Run with:  python trainer_test.py
"""

import os
import tempfile
from unittest.mock import patch

import torch

from configuration import Config
from maskgit_model import MaskedViT
from trainer import Trainer
from utils import step_schedule, make_masks

# ---------------------------------------------------------------------------
# Shared small config (fast to run, no GPU needed)
# ---------------------------------------------------------------------------

B = 2      # batch size
N = 16     # token grid: 4x4  (img_size=64, f=16 -> 64/16=4)
EMBED = 64
HEADS = 4
CODEBOOK = 8
CLIP_DIM = 32


def small_cfg(save_dir: str) -> Config:
    cfg            = Config()
    cfg.model.img_size      = 64
    cfg.model.embed_dim     = EMBED
    cfg.model.num_heads     = HEADS
    cfg.model.depth         = 2
    cfg.model.mlp_ratio     = 2.0
    cfg.model.codebook_size = CODEBOOK
    cfg.model.clip_dim      = CLIP_DIM
    cfg.model.attn_drop     = 0.0
    cfg.model.proj_drop     = 0.0
    cfg.train.device                 = "cpu"
    cfg.train.grad_clip              = 1.0
    cfg.train.next_index_loss_weight = 0.5
    cfg.train.log_interval           = 1
    cfg.train.checkpoint_interval    = 1
    cfg.train.save_dir               = save_dir
    cfg.train.epochs                 = 1
    return cfg


def make_trainer(cfg: Config) -> Trainer:
    model     = MaskedViT(cfg.model)
    dino_mock = None            # patched away in every test that calls train_step
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return Trainer(model, dino_mock, optimizer, cfg)


def random_sorted_idx(b: int, n: int) -> torch.Tensor:
    """Random (B, N) permutation — stands in for DINO output."""
    return torch.stack([torch.randperm(n) for _ in range(b)])


def schedule_for_n(n: int) -> list:
    T = max(1, int(torch.log2(torch.tensor(n, dtype=torch.float)).item()))
    return step_schedule(T).tolist()


# ---------------------------------------------------------------------------
# _make_masks
# ---------------------------------------------------------------------------

def test_make_masks_shapes():
    schedule   = schedule_for_n(N)
    sorted_idx = random_sorted_idx(B, N)
    step_idx   = torch.randint(0, len(schedule), (B,))
    q, k, nq   = make_masks(sorted_idx, schedule, step_idx, N, B)

    assert q.shape  == (B, N), f"query_mask shape: {q.shape}"
    assert k.shape  == (B, N), f"key_mask shape:   {k.shape}"
    assert nq.shape == (B, N), f"next_query shape:  {nq.shape}"
    print("[PASS] make_masks shapes")


def test_make_masks_disjoint():
    """query_mask, key_mask, next_query must be pairwise disjoint."""
    schedule   = schedule_for_n(N)
    sorted_idx = random_sorted_idx(B, N)
    step_idx   = torch.zeros(B, dtype=torch.long)
    q, k, nq   = make_masks(sorted_idx, schedule, step_idx, N, B)

    assert not (q & k).any(),  "query_mask and key_mask overlap"
    assert not (q & nq).any(), "query_mask and next_query overlap"
    assert not (k & nq).any(), "key_mask and next_query overlap"
    print("[PASS] make_masks disjoint")


def test_make_masks_step0_no_key():
    """At step 0 there are no previously-revealed tokens: key_mask must be empty."""
    schedule   = schedule_for_n(N)
    sorted_idx = random_sorted_idx(B, N)
    step_idx   = torch.zeros(B, dtype=torch.long)
    _, k, _    = make_masks(sorted_idx, schedule, step_idx, N, B)

    assert not k.any(), "key_mask should be empty at step 0"
    print("[PASS] make_masks step-0 key_mask is empty")


def test_make_masks_coverage():
    """query + key + next_query should not exceed N tokens per image."""
    schedule   = schedule_for_n(N)
    sorted_idx = random_sorted_idx(B, N)
    step_idx   = torch.randint(0, len(schedule), (B,))
    q, k, nq   = make_masks(sorted_idx, schedule, step_idx, N, B)

    for b in range(B):
        total = int((q[b] | k[b] | nq[b]).sum().item())
        assert total <= N, f"masks cover {total} > {N} tokens for image {b}"
    print("[PASS] make_masks total coverage <= N")


# ---------------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------------

def test_train_step_returns_losses():
    """train_step should return two finite scalar losses."""
    with tempfile.TemporaryDirectory() as d:
        cfg     = small_cfg(d)
        trainer = make_trainer(cfg)
        schedule = schedule_for_n(N)

        codes     = torch.randint(0, CODEBOOK, (B, N))
        images    = torch.rand(B, 3, 64, 64)
        clip_feat = torch.rand(B, CLIP_DIM)
        step_idx  = torch.randint(1, len(schedule), (B,))  # start at ≥1 for key_mask

        sorted_idx = random_sorted_idx(B, N)

        with patch("trainer.get_patch_sorted_index", return_value=sorted_idx):
            ce, bce, acc = trainer.train_step(codes, images, clip_feat, schedule, step_idx)

    assert isinstance(ce, float) and torch.isfinite(torch.tensor(ce)), f"CE not finite: {ce}"
    assert isinstance(bce, float) and torch.isfinite(torch.tensor(bce)), f"BCE not finite: {bce}"
    assert 0.0 <= acc <= 1.0, f"accuracy out of range: {acc}"
    print(f"[PASS] train_step losses  CE={ce:.4f}  BCE={bce:.4f}  Acc={acc:.3f}")


def test_train_step_updates_weights():
    """Model parameters must change after one train_step."""
    with tempfile.TemporaryDirectory() as d:
        cfg     = small_cfg(d)
        trainer = make_trainer(cfg)
        schedule = schedule_for_n(N)

        before = {n: p.clone() for n, p in trainer.model.named_parameters()}

        codes     = torch.randint(0, CODEBOOK, (B, N))
        images    = torch.rand(B, 3, 64, 64)
        clip_feat = torch.rand(B, CLIP_DIM)
        step_idx  = torch.randint(1, len(schedule), (B,))
        sorted_idx = random_sorted_idx(B, N)

        with patch("trainer.get_patch_sorted_index", return_value=sorted_idx):
            trainer.train_step(codes, images, clip_feat, schedule, step_idx)

    changed = any(
        not torch.equal(p, before[n])
        for n, p in trainer.model.named_parameters()
        if p.requires_grad
    )
    assert changed, "No parameters were updated after train_step"
    print("[PASS] train_step updates model parameters")


def test_train_step_step0_bce_nonzero():
    """At step 0, next_query is non-empty so BCE loss should be finite and non-zero."""
    with tempfile.TemporaryDirectory() as d:
        cfg     = small_cfg(d)
        trainer = make_trainer(cfg)
        schedule = schedule_for_n(N)

        codes     = torch.randint(0, CODEBOOK, (B, N))
        images    = torch.rand(B, 3, 64, 64)
        clip_feat = torch.rand(B, CLIP_DIM)
        step_idx  = torch.zeros(B, dtype=torch.long)   # step 0
        sorted_idx = random_sorted_idx(B, N)

        with patch("trainer.get_patch_sorted_index", return_value=sorted_idx):
            _, bce, _ = trainer.train_step(codes, images, clip_feat, schedule, step_idx)

    assert torch.isfinite(torch.tensor(bce)) and bce > 0.0, \
        f"BCE should be finite and non-zero at step 0, got {bce}"
    print(f"[PASS] train_step step-0 BCE is non-zero ({bce:.4f})")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def test_save_and_load_checkpoint():
    """Checkpoint round-trip: weights loaded into a fresh model must match original."""
    with tempfile.TemporaryDirectory() as d:
        cfg     = small_cfg(d)
        trainer = make_trainer(cfg)

        # Run one step so weights are non-trivial
        schedule   = schedule_for_n(N)
        codes      = torch.randint(0, CODEBOOK, (B, N))
        images     = torch.rand(B, 3, 64, 64)
        clip_feat  = torch.rand(B, CLIP_DIM)
        step_idx   = torch.randint(1, len(schedule), (B,))
        sorted_idx = random_sorted_idx(B, N)

        with patch("trainer.get_patch_sorted_index", return_value=sorted_idx):
            trainer.train_step(codes, images, clip_feat, schedule, step_idx)

        path = trainer.save_checkpoint(epoch=3, step=42)
        assert os.path.exists(path), f"Checkpoint file not created: {path}"

        # Load into a brand-new model instance
        fresh_model     = MaskedViT(cfg.model)
        fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
        fresh_trainer   = Trainer(fresh_model, None, fresh_optimizer, cfg)

        epoch, step = fresh_trainer.load_checkpoint(path)
        assert epoch == 3 and step == 42, f"epoch/step mismatch: {epoch}, {step}"

        # Every parameter in fresh model must equal the original
        orig  = dict(trainer.model.named_parameters())
        fresh = dict(fresh_trainer.model.named_parameters())
        for name in orig:
            assert torch.equal(orig[name], fresh[name]), \
                f"Parameter '{name}' mismatch after load"

    print("[PASS] save / load checkpoint round-trip")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_make_masks_shapes()
    test_make_masks_disjoint()
    test_make_masks_step0_no_key()
    test_make_masks_coverage()
    test_train_step_returns_losses()
    test_train_step_updates_weights()
    test_train_step_step0_bce_nonzero()
    test_save_and_load_checkpoint()
    print("\nAll trainer tests passed.")
