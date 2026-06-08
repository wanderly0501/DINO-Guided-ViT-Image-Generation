"""
MaskGIT inference and visualization utilities.

Generation:
  - Starts with all N tokens masked.
  - At each step t the schedule reveals 2^t tokens cumulatively.
  - pred_next scores are used to pick WHICH masked positions to reveal.
  - logits (with temperature) determine WHAT token to place there.
  - Final tokens are decoded by the VQ-GAN model into an image.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from maskgit_model import MaskedViT
from utils import step_schedule, visualize_steps


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model:        MaskedViT,
    vq_model,                        # PretrainedTokenizer with .decode(tokens)
    clip_feat:    torch.Tensor,      # (B, clip_dim) CLIP class embedding
    N:            int   = 256,       # total tokens (16x16 grid for f=16)
    T:            int   = 8,         # steps: schedule produces [1,2,4,...,2^T]
    temperature:  float = 1.0,       # sampling temperature for logits
    device:       str   = "cpu",
    visualize:    bool  = False,     # show per-step snapshots (batch item 0 only)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Iteratively unmask tokens to generate an image.

    Returns:
        images: (B, 3, H, W)  decoded images in [0, 1]
        tokens: (B, N)        final token sequence
    """
    model.eval()
    clip_feat = clip_feat.to(device)
    B = clip_feat.shape[0]

    MASK_ID   = model.cfg.codebook_size   # [MASK] token index
    schedule  = step_schedule(T)          # (T+1,) = [1, 2, 4, ..., 2^T]
    snapshots = [] if visualize else None

    # Start: all tokens masked
    tokens       = torch.full((B, N), MASK_ID, dtype=torch.long, device=device)
    revealed     = torch.zeros(B, N, dtype=torch.bool, device=device)
    reveal_step  = torch.full((B, N), -1, dtype=torch.long, device=device)

    # Initial query_mask: randomly pick schedule[0] positions, avoiding patches
    # within 2 of any edge so generation starts away from borders.
    grid = int(N ** 0.5)
    all_idx  = torch.arange(N, device=device)
    rows     = all_idx // grid
    cols     = all_idx % grid
    interior = all_idx[(rows >= 2) & (rows < grid - 2) & (cols >= 2) & (cols < grid - 2)]

    n_first    = min(int(schedule[0].item()), len(interior))
    query_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    for b in range(B):
        chosen = interior[torch.randperm(len(interior), device=device)[:n_first]]
        query_mask[b, chosen] = True

    for t, n_total in enumerate(schedule.tolist()):
        key_mask    = revealed.clone()                                  # (B, N)
        step_tensor = torch.full((B,), t, dtype=torch.long, device=device)

        logits, pred_next = model(tokens, query_mask, key_mask, clip_feat, step=step_tensor)
        # logits:    (B, N, codebook_size)
        # pred_next: (B, N, 1)

        # 1. Use logits to sample tokens at query_mask positions
        for b in range(B):
            for pos in query_mask[b].nonzero(as_tuple=True)[0].tolist():
                probs               = F.softmax(logits[b, pos] / temperature, dim=-1)
                tokens[b, pos]      = torch.multinomial(probs, 1).item()
                revealed[b, pos]    = True
                reveal_step[b, pos] = t

        # 2. Use top-k pred_next scores as query_mask for next step
        if t + 1 < len(schedule):
            n_next     = min(int(schedule[t + 1].item()), N) - min(int(n_total), N)
            query_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
            if n_next > 0:
                scores     = pred_next.squeeze(-1)                      # (B, N)
                scores     = scores.masked_fill(revealed, float("-inf"))
                _, top_idx = scores.topk(n_next, dim=1)                # (B, n_next)
                for b in range(B):
                    query_mask[b, top_idx[b]] = True

        if snapshots is not None:
            partial = tokens[0:1].clone()
            partial[~revealed[0:1]] = 0
            img = vq_model.decode(partial).squeeze(0).cpu()            # (3, H, W)
            snapshots.append((img, f"step {t}\n{int(revealed[0].sum())} revealed"))

    if snapshots is not None:
        visualize_steps(snapshots)

    # Clamp any unfilled MASK_ID tokens to the valid codebook range before decode
    tokens = tokens.clamp(max=MASK_ID - 1)

    # Decode final tokens with VQ-GAN
    images = vq_model.decode(tokens)    # (B, 3, H, W) in [0, 1]
    return images, tokens, reveal_step


@torch.no_grad()
def generate_from_partial(
    model:              MaskedViT,
    vq_model,
    clip_feat:          torch.Tensor,   # (B, clip_dim)
    context_codes:      torch.Tensor,   # (B, N) GT token ids; only key_mask positions used
    key_mask:           torch.Tensor,   # (B, N) bool — already-revealed context
    initial_query_mask: torch.Tensor,   # (B, N) bool — first batch to predict (step start_step)
    start_step:         int,            # step index to resume from
    N:                  int   = 256,
    T:                  int   = 8,
    temperature:        float = 1.0,
    device:             str   = "cpu",
    greedy:             bool  = False,  # if True, argmax instead of sampling
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Complete generation starting from a partially revealed image.

    Returns:
        images:      (B, 3, H, W)  decoded in [0, 1]
        tokens:      (B, N)        final token sequence
        reveal_step: (B, N)        step each token was revealed (-1 = given context)
    """
    model.eval()
    clip_feat     = clip_feat.to(device)
    context_codes = context_codes.to(device)
    key_mask      = key_mask.to(device)

    B       = clip_feat.shape[0]
    MASK_ID = model.cfg.codebook_size
    schedule = step_schedule(T)

    tokens      = torch.full((B, N), MASK_ID, dtype=torch.long, device=device)
    tokens[key_mask] = context_codes[key_mask]
    revealed    = key_mask.clone()
    reveal_step = torch.full((B, N), -1, dtype=torch.long, device=device)

    query_mask = initial_query_mask.to(device)

    for t in range(start_step, len(schedule)):
        n_total     = int(schedule[t].item())
        step_tensor = torch.full((B,), t, dtype=torch.long, device=device)

        logits, pred_next = model(tokens, query_mask, revealed.clone(), clip_feat, step=step_tensor)

        for b in range(B):
            for pos in query_mask[b].nonzero(as_tuple=True)[0].tolist():
                if greedy:
                    tokens[b, pos] = logits[b, pos].argmax(dim=-1).item()
                else:
                    probs          = F.softmax(logits[b, pos] / temperature, dim=-1)
                    tokens[b, pos] = torch.multinomial(probs, 1).item()
                revealed[b, pos]    = True
                reveal_step[b, pos] = t

        if t + 1 < len(schedule):
            n_next     = min(int(schedule[t + 1].item()), N) - min(n_total, N)
            query_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
            if n_next > 0:
                scores     = pred_next.squeeze(-1).masked_fill(revealed, float("-inf"))
                _, top_idx = scores.topk(n_next, dim=1)
                for b in range(B):
                    query_mask[b, top_idx[b]] = True

    tokens = tokens.clamp(max=MASK_ID - 1)
    images = vq_model.decode(tokens)
    return images, tokens, reveal_step



