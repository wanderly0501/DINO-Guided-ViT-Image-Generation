"""
Pre-training entry point for the MaskGIT model using random masking.

Usage:
  # New model
  python pre_main.py

  # Resume from checkpoint
  python pre_main.py --checkpoint checkpoints/pretrain_epoch0000_step000100.pt
"""

import argparse
import sys
import os
import torch
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="Corrupt EXIF data", category=UserWarning)
import clip

import importlib

from configuration import Config
from maskgit_model import MaskedViT
from data import get_dataloaders, get_hf_dataloaders, get_imagenet_class_names
from utils import denormalize

# "pre-training.py" has a hyphen, so normal `from pre-training import ...` is a
# syntax error. importlib handles the hyphenated module name correctly.
_pt_module = importlib.import_module("pre-training")
RandomMaskTrainer = _pt_module.RandomMaskTrainer

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "1d-tokenizer"))
from modeling.titok import PretrainedTokenizer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-train MaskGIT with random masking")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a .pt checkpoint to resume pre-training from. "
             "If omitted, a new model is initialised from scratch.",
    )
    parser.add_argument(
        "--ckpt_path", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "ckpts", "maskgit-vqgan-imagenet-f16-256.bin"),
        help="Path to the pre-trained VQ-GAN tokenizer weights.",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Load ImageNet from local files instead of HuggingFace (ILSVRC/imagenet-1k).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model wrappers (identical to main.py)
# ---------------------------------------------------------------------------

class VQGANWrapper(torch.nn.Module):
    """Wraps PretrainedTokenizer so the trainer can call .encode(images)."""

    def __init__(self, tokenizer: PretrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        images_01 = denormalize(images)
        return self.tokenizer.encode(images_01)   # (B, N)

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.decode(tokens)       # (B, 3, H, W) in [0, 1]


class CLIPWrapper(torch.nn.Module):
    """Wraps an OpenAI CLIP model so the trainer can call .encode_text(labels)."""

    def __init__(self, clip_model, class_names: list[str], device):
        super().__init__()
        self.clip_model  = clip_model
        self.class_names = class_names
        self.device      = device

    @torch.no_grad()
    def encode_text(self, labels: torch.Tensor) -> torch.Tensor:
        """labels: (B,) class indices → (B, clip_dim) float32"""
        prompts = clip.tokenize(
            [f"a photo of a {self.class_names[i]}" for i in labels.tolist()]
        ).to(self.device)
        return self.clip_model.encode_text(prompts).float()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = Config()
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- VQ-GAN tokenizer ---------------------------------------------------
    print("Loading VQ-GAN tokenizer ...")
    tokenizer = PretrainedTokenizer(args.ckpt_path)
    vq_model  = VQGANWrapper(tokenizer).to(device)

    # --- CLIP model (class conditioning) ------------------------------------
    print("Loading CLIP model ...")
    clip_model_raw, _ = clip.load("ViT-B/32", device=device)
    clip_model_raw.eval()
    clip_model = None   # initialised after dataloaders (needs class_names)

    # --- MaskGIT model ------------------------------------------------------
    model    = MaskedViT(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MaskedViT parameters: {n_params:,}")

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
        model.cuda()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    trainer = RandomMaskTrainer(model, optimizer, cfg)

    start_epoch = 0
    if args.checkpoint is not None:
        print(f"Resuming from checkpoint: {args.checkpoint}")
        start_epoch, _ = trainer.load_checkpoint(args.checkpoint)
        start_epoch += 1
    else:
        print("Initialising new model from scratch.")

    # --- Data ---------------------------------------------------------------
    print("Building dataloaders ...")
    if args.local:
        train_loader, val_loader = get_dataloaders(
            dataset_dir=cfg.data.dataset_dir,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            img_size=cfg.data.img_size,
        )
    else:
        train_loader, val_loader, _ = get_hf_dataloaders(
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            img_size=cfg.data.img_size,
        )

    class_names = get_imagenet_class_names()
    clip_model  = CLIPWrapper(clip_model_raw, class_names, device)

    # --- Pre-training -------------------------------------------------------
    print("Starting pre-training ...")
    trainer.train(
        dataloader=train_loader,
        val_loader=val_loader,
        vq_model=vq_model,
        clip_model=clip_model,
    )


if __name__ == "__main__":
    main()
