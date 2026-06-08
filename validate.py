import os
import torch
import torch.nn.functional as F

from maskgit_model import MaskedViT
from utils import step_schedule, make_masks, visualize_steps, mask_to_image
from dino_util import get_patch_sorted_index
from configuration import Config


@torch.no_grad()
def validate(
    model:      MaskedViT,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    vq_model,
    clip_model,
    dino_model,
    cfg:        Config,
    epoch:      int = 0,
    step:       int = 0,
) -> tuple[float, float, float]:
    """Visualize model predictions on val images at random steps.

    For each sampled image shows three panels side by side:
      [masked input | model prediction | ground truth]
    """
    model.eval()
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    images = val_images[:cfg.train.val_num_samples].to(device)
    labels = val_labels[:cfg.train.val_num_samples].to(device)
    B = images.shape[0]

    N        = (cfg.model.img_size // 16) ** 2
    T        = max(1, int(torch.log2(torch.tensor(N, dtype=torch.float)).item()))
    schedule = step_schedule(T).tolist()
    MASK_ID  = cfg.model.codebook_size

    codes     = vq_model.encode(images)           # (B, N)
    clip_feat = clip_model.encode_text(labels)    # (B, clip_dim)
    sorted_idx = get_patch_sorted_index(images, dino_model).to(device)  # (B, N)

    step_idx = torch.randint(0, T + 1, (B,), device=device)
    query_mask, key_mask, next_query = make_masks(sorted_idx, schedule, step_idx, N, B, device=device)

    masked_codes = codes.clone()
    masked_codes[~key_mask] = MASK_ID

    logits, pred_next = model(masked_codes, query_mask, key_mask, clip_feat, step=step_idx)
    # logits:    (B, N, codebook_size)
    # pred_next: (B, N, 1)

    pred_tokens = logits.argmax(dim=-1)                    # (B, N)

    # -- CE loss and accuracy on query positions -----------------------------
    T          = len(schedule) - 1
    ce_logits  = logits[query_mask]
    ce_targets = codes[query_mask]
    accuracy   = (pred_tokens[query_mask] == codes[query_mask]).float().mean().item()

    if cfg.train.per_example_loss:
        q_batch   = query_mask.nonzero(as_tuple=True)[0]
        ce_tokens = F.cross_entropy(ce_logits, ce_targets, reduction='none')
        q_counts  = query_mask.sum(dim=1).float()
        ce_sum    = torch.zeros(B, device=device).scatter_add_(0, q_batch, ce_tokens)
        ce_loss   = (ce_sum / q_counts.clamp(min=1))[q_counts > 0].mean()
    else:
        ce_loss = F.cross_entropy(ce_logits, ce_targets)

    # -- BCE loss on remaining masked positions ------------------------------
    bce_valid = (step_idx < T - 1).unsqueeze(1).expand(B, N)
    remaining = ~(query_mask | key_mask) & bce_valid
    if remaining.any():
        bce_pred   = pred_next[remaining].squeeze(-1)
        bce_target = next_query[remaining].float()
        if cfg.train.per_example_loss:
            r_batch  = remaining.nonzero(as_tuple=True)[0]
            bce_tok  = F.binary_cross_entropy_with_logits(bce_pred, bce_target, reduction='none')
            r_counts = remaining.sum(dim=1).float()
            bce_sum  = torch.zeros(B, device=device).scatter_add_(0, r_batch, bce_tok)
            bce_loss = (bce_sum / r_counts.clamp(min=1))[r_counts > 0].mean()
        else:
            bce_loss = F.binary_cross_entropy_with_logits(bce_pred, bce_target)
    else:
        bce_loss = torch.tensor(0.0, device=device)

    print(f"Val | Epoch {epoch:3d} | Step {step:5d} "
          f"| CE {ce_loss:.4f} | BCE {bce_loss:.4f} | Acc {accuracy:.3f}")
    predicted_codes = codes.clone()
    predicted_codes[query_mask]              = pred_tokens[query_mask]
    predicted_codes[~(key_mask | query_mask)] = 0          # hide fully-masked positions

    for b in range(B):
        masked_display = masked_codes[b:b+1].clone()
        masked_display[masked_display == MASK_ID] = 0      # replace [MASK] for decode

        img_masked = vq_model.decode(masked_display).squeeze(0).cpu()
        img_pred   = vq_model.decode(predicted_codes[b:b+1]).squeeze(0).cpu()
        img_gt     = vq_model.decode(codes[b:b+1]).squeeze(0).cpu()

        t         = int(step_idx[b].item())
        n_context = int(key_mask[b].sum().item())
        n_query   = int(query_mask[b].sum().item())
        n_next    = int(next_query[b].sum().item())

        heatmap = mask_to_image(key_mask[b], query_mask[b], next_query[b])  # (3, grid, grid)
        legend  = (
            f"mask legend\n"
            f"green={n_context} ctx  gold={n_query} query\n"
            f"blue={n_next} next  gray=masked"
        )

        save_path = os.path.join(
            cfg.train.output_dir, "val",
            f"epoch{epoch:04d}_step{step:06d}_sample{b}.png",
        )
        class_name = clip_model.class_names[labels[b].item()].split(",")[0]
        visualize_steps(
            [
                (img_masked, f"masked input\n{n_context} context tokens"),
                (img_pred,   f"model output\nstep {t}, +{n_query} tokens"),
                (img_gt,     f"ground truth\n{class_name}"),
                (heatmap,    legend),
            ],
            title=f"Validation sample {b}  —  {class_name}  (step {t})",
            save_path=save_path,
        )

    return ce_loss.item(), bce_loss.item(), accuracy
