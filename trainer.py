import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from configuration import Config
from maskgit_model import MaskedViT
from dino_util import get_patch_sorted_index
from utils import step_schedule, make_masks
from validate import validate


class Trainer:
    """MaskGIT-style trainer.

    Each batch item gets a random step index t drawn from [0, T].
    key_mask:   first schedule[t-1] tokens in DINO order (previous-step context).
    query_mask: tokens schedule[t-1]..schedule[t] in DINO order (predicted now).
    Tokens beyond schedule[t] are fully masked but not in either mask.
    A single forward pass is made per batch.
    """

    def __init__(
        self,
        model:      MaskedViT,
        dino_model: nn.Module,
        optimizer:  torch.optim.Optimizer,
        cfg:        Config,
    ):
        self.model      = model
        self.dino_model = dino_model
        self.optimizer  = optimizer
        self.cfg        = cfg
        self.device     = torch.device(cfg.train.device
                                       if torch.cuda.is_available() else "cpu")
        self.use_bf16   = cfg.train.use_bf16 and torch.cuda.is_available()
        self.scheduler  = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Single training step over one batch
    # ------------------------------------------------------------------

    def train_step(
        self,
        codes:     torch.Tensor,   # (B, N)  true VQ-GAN token ids
        images:    torch.Tensor,   # (B, 3, H, W)
        clip_feat: torch.Tensor,   # (B, clip_dim)
        schedule:  list,           # precomputed in train()
        step_idx:  torch.Tensor,   # (B,)  random step index per image in [0, T]
    ):
        B, N = codes.shape
        codes     = codes.to(self.device)
        images    = images.to(self.device)
        clip_feat = clip_feat.to(self.device)

        # DINO-sorted patch order for every image: (B, N)
        with torch.no_grad():
            sorted_idx = get_patch_sorted_index(images, self.dino_model)
        sorted_idx = sorted_idx.to(self.device)

        # Build masks using each image's own random step
        query_mask, key_mask, next_query = make_masks(
            sorted_idx, schedule, step_idx, N, B, device=self.device
        )

        # Replace all non-context positions with [MASK] token id
        masked_codes = codes.clone()
        masked_codes[~key_mask] = self.cfg.model.codebook_size

        # Single forward pass + loss under bf16 autocast
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            logits, pred_next = self.model(
                masked_codes, query_mask, key_mask, clip_feat, step=step_idx
            )
            # logits:    (B, N, codebook_size)
            # pred_next: (B, N, 1)

            # -- CE loss and accuracy on query positions ----------------------
            T          = len(schedule) - 1
            ce_logits  = logits[query_mask]       # (Q, codebook_size)
            ce_targets = codes[query_mask]        # (Q,)
            accuracy   = (ce_logits.argmax(dim=-1) == ce_targets).float().mean().item()

            if self.cfg.train.per_example_loss:
                q_batch    = query_mask.nonzero(as_tuple=True)[0]
                ce_tokens  = F.cross_entropy(ce_logits, ce_targets, reduction='none')
                q_counts   = query_mask.sum(dim=1).float()
                ce_sum     = torch.zeros(B, device=self.device).scatter_add_(0, q_batch, ce_tokens)
                ce_loss    = (ce_sum / q_counts.clamp(min=1))[q_counts > 0].mean()
            else:
                ce_loss    = F.cross_entropy(ce_logits, ce_targets)

            # -- BCE loss: over all remaining masked positions ----------------
            bce_valid = (step_idx < T - 1).unsqueeze(1).expand(B, N)
            remaining = ~(query_mask | key_mask) & bce_valid
            if remaining.any():
                bce_pred   = pred_next[remaining].squeeze(-1)
                bce_target = next_query[remaining].float()
                if self.cfg.train.per_example_loss:
                    r_batch   = remaining.nonzero(as_tuple=True)[0]
                    bce_tok   = F.binary_cross_entropy_with_logits(bce_pred, bce_target, reduction='none')
                    r_counts  = remaining.sum(dim=1).float()
                    bce_sum   = torch.zeros(B, device=self.device).scatter_add_(0, r_batch, bce_tok)
                    bce_loss  = (bce_sum / r_counts.clamp(min=1))[r_counts > 0].mean()
                else:
                    bce_loss  = F.binary_cross_entropy_with_logits(bce_pred, bce_target)
            else:
                bce_loss = torch.tensor(0.0, device=self.device)

            loss = ce_loss + self.cfg.train.next_index_loss_weight * bce_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.train.grad_clip
        )
        self.optimizer.step()

        return ce_loss.item(), bce_loss.item(), accuracy

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, step: int) -> str:
        """Save model + optimizer state to cfg.train.save_dir.

        Returns the path of the saved file.
        """
        save_dir = self.cfg.train.save_dir
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"ckpt_epoch{epoch:04d}_step{step:06d}.pt")
        raw_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        torch.save(
            {
                "epoch":             epoch,
                "step":              step,
                "model_state":       raw_model.state_dict(),
                "optimizer_state":   self.optimizer.state_dict(),
                "scheduler_state":   self.scheduler.state_dict() if self.scheduler else None,
            },
            path,
        )
        print(f"Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path: str) -> tuple[int, int]:
        """Load model + optimizer state from a checkpoint file.

        Returns (epoch, step) so training can resume from the right position.
        """
        checkpoint = torch.load(path, map_location=self.device)
        raw_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        raw_model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        epoch = checkpoint.get("epoch", 0)
        step  = checkpoint.get("step", 0)
        print(f"Checkpoint loaded: {path}  (epoch={epoch}, step={step})")
        return epoch, step

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        dataloader,
        val_loader,
        vq_model:   nn.Module,   # PretrainedTokenizer -- encode returns (B, N)
        clip_model,              # OpenAI CLIP model
    ):
        self.model.train()
        cfg = self.cfg

        # may need to change if train on cloud
        os.makedirs(cfg.train.output_dir, exist_ok=True)
        loss_log_path = os.path.join(cfg.train.output_dir, "losses.csv")
        if not os.path.exists(loss_log_path):
            with open(loss_log_path, "w") as f:
                f.write("epoch,step,split,ce_loss,bce_loss,accuracy\n")

        # Compute schedule once
        N        = (cfg.model.img_size // 16) ** 2   # 256 for 256-px images
        T        = max(1, int(torch.log2(torch.tensor(N, dtype=torch.float)).item()))
        schedule = step_schedule(T).tolist()          # [1, 2, 4, ..., 2^T]

        # Build step-based LR scheduler now that we know len(dataloader)
        total_steps  = cfg.train.epochs * len(dataloader)
        warmup_steps = cfg.train.warmup_epochs * len(dataloader)
        eta_min_factor = 1e-6 / cfg.train.learning_rate

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return eta_min_factor + (1.0 - eta_min_factor) * current_step / max(warmup_steps, 1)
            progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return eta_min_factor + (1.0 - eta_min_factor) * 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        val_iter = enumerate(val_loader)

        for epoch in range(cfg.train.epochs):
            for step, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():
                    codes     = vq_model.encode(images)          # (B, N)
                    clip_feat = clip_model.encode_text(labels)   # (B, clip_dim)

                B = codes.shape[0]
                # random step index per image: (B,) each in [0, T]
                step_idx = torch.randint(0, T + 1, (B,), device=self.device)

                ce_loss, bce_loss, accuracy = self.train_step(
                    codes, images, clip_feat, schedule, step_idx
                )
                self.scheduler.step()

                if step % cfg.train.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    print(
                        f"Epoch {epoch:3d} | Step {step:5d} "
                        f"| CE {ce_loss:.4f} | BCE {bce_loss:.4f} | Acc {accuracy:.3f} | LR {lr:.2e}"
                    )
                    with open(loss_log_path, "a") as f:
                        f.write(f"{epoch},{step},train,{ce_loss:.6f},{bce_loss:.6f},{accuracy:.6f}\n")

                if step % cfg.train.checkpoint_interval == 0 or step == len(dataloader) - 1:
                    self.save_checkpoint(epoch, step)

                if step % cfg.train.val_interval == 0:
                    try:
                        _, (val_images, val_labels) = next(val_iter)
                    except StopIteration:
                        val_iter = enumerate(val_loader)
                        _, (val_images, val_labels) = next(val_iter)
                    val_ce, val_bce, val_acc = validate(
                        self.model, val_images, val_labels, vq_model, clip_model, self.dino_model,
                        self.cfg, epoch=epoch, step=step,
                    )
                    with open(loss_log_path, "a") as f:
                        f.write(f"{epoch},{step},val,{val_ce:.6f},{val_bce:.6f},{val_acc:.6f}\n")
                    self.model.train()
