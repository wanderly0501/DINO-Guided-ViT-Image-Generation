"""
Main training entry point for the MaskGIT model.

Usage:
  # New model
  python main.py

  # Resume from checkpoint
  python main.py --checkpoint checkpoints/ckpt_epoch0000_step000100.pt
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

from configuration import Config
from maskgit_model import MaskedViT
from trainer import Trainer
from data import get_dataloaders, get_hf_dataloaders, get_imagenet_class_names
from dino_util import load_dino_model
from utils import denormalize

# Add 1d-tokenizer repo to path for PretrainedTokenizer
sys.path.append(os.path.join(os.path.dirname(__file__), "1d-tokenizer"))
from modeling.titok import PretrainedTokenizer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MaskGIT image generation model")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a .pt checkpoint to resume training from. "
             "If omitted, a new model is initialised from scratch.",
    )
    parser.add_argument(
        "--ckpt_path", type=str,
        default=os.path.join(os.path.dirname(__file__), "ckpts", "maskgit-vqgan-imagenet-f16-256.bin"),
        help="Path to the pre-trained VQ-GAN tokenizer weights.",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Load ImageNet from local files instead of HuggingFace (ILSVRC/imagenet-1k).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model wrappers that expose a unified .encode() API for the Trainer
# ---------------------------------------------------------------------------

class VQGANWrapper(torch.nn.Module):
    """Wraps PretrainedTokenizer so Trainer can call .encode(images)."""

    def __init__(self, tokenizer: PretrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 3, H, W) ImageNet-normalised in [-1, 1] range
                (already converted from ImageNet stats by caller)
        Returns tokens: (B, N) int64
        """
        images_01 = denormalize(images)          # [-1,1] -> [0,1]
        return self.tokenizer.encode(images_01)  # (B, N)

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, N) int64
        Returns: (B, 3, H, W) in [0, 1]
        """
        return self.tokenizer.decode(tokens)


class CLIPWrapper(torch.nn.Module):
    """Wraps an OpenAI CLIP model so Trainer can call .encode_text(labels)."""

    def __init__(self, clip_model, class_names: list[str], device):
        super().__init__()
        self.clip_model  = clip_model
        self.class_names = class_names
        self.device      = device

    @torch.no_grad()
    def encode_text(self, labels: torch.Tensor) -> torch.Tensor:
        """
        labels: (B,) class indices
        Returns clip_feat: (B, clip_dim) float32
        """
        # print("prompts: ", [self.class_names[i] for i in labels.tolist()])
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

    # --- DINO model (patch ordering) ----------------------------------------
    print("Loading DINO model ...")
    dino_model = load_dino_model(arch=cfg.dino.arch)
    dino_model = dino_model.to(device).eval()

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
    trainer = Trainer(model, dino_model, optimizer, cfg)

    start_epoch = 0
    if args.checkpoint is not None:
        print(f"Resuming from checkpoint: {args.checkpoint}")
        start_epoch, _ = trainer.load_checkpoint(args.checkpoint)
        start_epoch += 1   # continue from the next epoch
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

    # --- Training -----------------------------------------------------------
    print("Starting training ...")
    trainer.train(
        dataloader=train_loader,
        val_loader=val_loader,
        vq_model=vq_model,
        clip_model=clip_model,
    )

    if cfg.train.train_predict_next:
        print("Fine-tuning PredictNext head (all other parameters frozen) ...")
        trainer.train_predict_next(
            dataloader=train_loader,
            val_loader=val_loader,
            vq_model=vq_model,
            clip_model=clip_model,
        )


if __name__ == "__main__":
    main()
