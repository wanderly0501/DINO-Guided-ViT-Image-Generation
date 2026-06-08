import os
import math
import torch
import torch.nn.functional as F

from configuration import Config
from maskgit_model import MaskedViT


class RandomMaskTrainer:
    """Pre-trainer for MaskGIT using purely random masking.

    Each batch item gets k ~ Uniform(1, N) patches masked at random (1–100 %
    of patches). All masked tokens are predicted via CE loss. No DINO ordering,
    no schedule, no BCE 'predict-next' auxiliary loss.
    """

    def __init__(
        self,
        model:     MaskedViT,
        optimizer: torch.optim.Optimizer,
        cfg:       Config,
    ):
        self.model     = model
        self.optimizer = optimizer
        self.cfg       = cfg
        self.device    = torch.device(
            cfg.train.device if torch.cuda.is_available() else "cpu"
        )
        self.use_bf16  = cfg.train.use_bf16 and torch.cuda.is_available()
        self.scheduler = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _random_masks(self, B: int, N: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (query_mask, key_mask), each (B, N) bool.

        For each image b, k_b ~ Uniform(1, N) patches are chosen randomly
        as the masked (query) set; the rest are context (key).
        """
        query_mask = torch.zeros(B, N, dtype=torch.bool, device=self.device)
        for b in range(B):
            k = torch.randint(1, N + 1, (1,)).item()
            masked_idx = torch.randperm(N, device=self.device)[:k]
            query_mask[b, masked_idx] = True
        key_mask = ~query_mask
        return query_mask, key_mask

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        codes:     torch.Tensor,   # (B, N)  true VQ-GAN token ids
        clip_feat: torch.Tensor,   # (B, clip_dim)
    ) -> tuple[float, float]:
        B, N = codes.shape
        codes     = codes.to(self.device)
        clip_feat = clip_feat.to(self.device)

        query_mask, key_mask = self._random_masks(B, N)

        # Replace masked positions with [MASK] token id
        masked_codes = codes.clone()
        masked_codes[query_mask] = self.cfg.model.codebook_size

        # step=0 for all: no unmasking-schedule concept in random pre-training
        step_idx = torch.zeros(B, dtype=torch.long, device=self.device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            logits, _ = self.model(masked_codes, query_mask, key_mask, clip_feat, step=step_idx)
            # logits: (B, N, codebook_size)

            # CE loss only on masked (query) positions
            ce_logits  = logits[query_mask]    # (Q, codebook_size)
            ce_targets = codes[query_mask]     # (Q,)

            accuracy = (ce_logits.argmax(dim=-1) == ce_targets).float().mean().item()

            if self.cfg.train.per_example_loss:
                q_batch   = query_mask.nonzero(as_tuple=True)[0]
                ce_tokens = F.cross_entropy(ce_logits, ce_targets, reduction='none')
                q_counts  = query_mask.sum(dim=1).float()
                ce_sum    = torch.zeros(B, device=self.device).scatter_add_(0, q_batch, ce_tokens)
                loss      = (ce_sum / q_counts.clamp(min=1)).mean()
            else:
                loss = F.cross_entropy(ce_logits, ce_targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
        self.optimizer.step()

        return loss.item(), accuracy

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, step: int) -> str:
        save_dir = self.cfg.train.save_dir
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"pretrain_epoch{epoch:04d}_step{step:06d}.pt")
        raw_model = (
            self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        )
        torch.save(
            {
                "epoch":           epoch,
                "step":            step,
                "model_state":     raw_model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            },
            path,
        )
        print(f"Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path: str) -> tuple[int, int]:
        """Load model + optimizer state.  Returns (epoch, step)."""
        checkpoint = torch.load(path, map_location=self.device)
        raw_model = (
            self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        )
        raw_model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        epoch = checkpoint.get("epoch", 0)
        step  = checkpoint.get("step", 0)
        print(f"Checkpoint loaded: {path}  (epoch={epoch}, step={step})")
        return epoch, step

    # ------------------------------------------------------------------
    # Validation (no DINO, no visualisation — metrics only)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(
        self,
        val_images: torch.Tensor,
        val_labels: torch.Tensor,
        vq_model,
        clip_model,
        epoch: int = 0,
        step:  int = 0,
    ) -> tuple[float, float]:
        self.model.eval()
        cfg     = self.cfg
        device  = self.device
        MASK_ID = cfg.model.codebook_size

        images = val_images[:cfg.train.val_num_samples].to(device)
        labels = val_labels[:cfg.train.val_num_samples].to(device)
        B = images.shape[0]

        codes     = vq_model.encode(images)         # (B, N)
        clip_feat = clip_model.encode_text(labels)  # (B, clip_dim)
        N = codes.shape[1]

        query_mask, key_mask = self._random_masks(B, N)

        masked_codes = codes.clone()
        masked_codes[query_mask] = MASK_ID

        step_idx = torch.zeros(B, dtype=torch.long, device=device)
        logits, _ = self.model(masked_codes, query_mask, key_mask, clip_feat, step=step_idx)

        ce_logits  = logits[query_mask]
        ce_targets = codes[query_mask]
        accuracy   = (ce_logits.argmax(dim=-1) == ce_targets).float().mean().item()

        if cfg.train.per_example_loss:
            q_batch   = query_mask.nonzero(as_tuple=True)[0]
            ce_tokens = F.cross_entropy(ce_logits, ce_targets, reduction='none')
            q_counts  = query_mask.sum(dim=1).float()
            ce_sum    = torch.zeros(B, device=device).scatter_add_(0, q_batch, ce_tokens)
            ce_loss   = (ce_sum / q_counts.clamp(min=1)).mean().item()
        else:
            ce_loss = F.cross_entropy(ce_logits, ce_targets).item()

        print(
            f"Val  | Epoch {epoch:3d} | Step {step:5d} "
            f"| CE {ce_loss:.4f} | Acc {accuracy:.3f}"
        )
        return ce_loss, accuracy

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        dataloader,
        val_loader,
        vq_model,    # PretrainedTokenizer — encode(images) → (B, N)
        clip_model,  # OpenAI CLIP model   — encode_text(labels) → (B, clip_dim)
    ):
        self.model.train()
        cfg = self.cfg

        os.makedirs(cfg.train.output_dir, exist_ok=True)
        loss_log_path = os.path.join(cfg.train.output_dir, "pretrain_losses.csv")
        if not os.path.exists(loss_log_path):
            with open(loss_log_path, "w") as f:
                f.write("epoch,step,split,ce_loss,accuracy\n")

        total_steps    = cfg.train.epochs * len(dataloader)
        warmup_steps   = cfg.train.warmup_epochs * len(dataloader)
        eta_min_factor = 1e-6 / cfg.train.learning_rate

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return (
                    eta_min_factor
                    + (1.0 - eta_min_factor) * current_step / max(warmup_steps, 1)
                )
            progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return eta_min_factor + (1.0 - eta_min_factor) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        val_iter = enumerate(val_loader)

        for epoch in range(cfg.train.epochs):
            for step, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():
                    codes     = vq_model.encode(images)         # (B, N)
                    clip_feat = clip_model.encode_text(labels)  # (B, clip_dim)

                ce_loss, accuracy = self.train_step(codes, clip_feat)
                self.scheduler.step()

                if step % cfg.train.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    print(
                        f"Epoch {epoch:3d} | Step {step:5d} "
                        f"| CE {ce_loss:.4f} | Acc {accuracy:.3f} | LR {lr:.2e}"
                    )
                    with open(loss_log_path, "a") as f:
                        f.write(f"{epoch},{step},train,{ce_loss:.6f},{accuracy:.6f}\n")

                if step % cfg.train.checkpoint_interval == 0 or step == len(dataloader) - 1:
                    self.save_checkpoint(epoch, step)

                if step % cfg.train.val_interval == 0:
                    try:
                        _, (val_images, val_labels) = next(val_iter)
                    except StopIteration:
                        val_iter = enumerate(val_loader)
                        _, (val_images, val_labels) = next(val_iter)
                    val_ce, val_acc = self._validate(
                        val_images, val_labels, vq_model, clip_model, epoch, step
                    )
                    with open(loss_log_path, "a") as f:
                        f.write(f"{epoch},{step},val,{val_ce:.6f},{val_acc:.6f}\n")
                    self.model.train()
