"""
Single train-step smoke test on GPU.

Usage:
  python main_test.py
  python main_test.py --ckpt_path ckpts/maskgit-vqgan-imagenet-f16-256.bin
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
from dino_util import load_dino_model
from utils import step_schedule
from data import get_hf_dataloaders, get_imagenet_class_names
from main import VQGANWrapper, CLIPWrapper

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "1d-tokenizer"))
from modeling.titok import PretrainedTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "ckpts", "maskgit-vqgan-imagenet-f16-256.bin"),
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.train.device = str(device)
    print(f"Device: {device}")

    # --- Models ---------------------------------------------------------------
    print("Loading VQ-GAN ...")
    vq_model = VQGANWrapper(PretrainedTokenizer(args.ckpt_path)).to(device)

    print("Loading DINO ...")
    dino_model = load_dino_model(arch=cfg.dino.arch).to(device).eval()

    # --- Real data ------------------------------------------------------------
    print("Loading one batch from train set ...")
    train_loader, _, _ = get_hf_dataloaders(
        batch_size=4,
        num_workers=0,
        img_size=cfg.model.img_size,
        pin_memory=False,
    )
    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)

    print("Loading CLIP ...")
    clip_raw, _ = clip.load("ViT-B/32", device=device)
    clip_raw.eval()
    clip_model = CLIPWrapper(clip_raw, get_imagenet_class_names(), device)

    # --- MaskGIT --------------------------------------------------------------
    model     = MaskedViT(cfg.model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.learning_rate,
                            weight_decay=cfg.train.weight_decay)
    trainer   = Trainer(model, dino_model, optimizer, cfg)

    B = images.shape[0]
    N = (cfg.model.img_size // 16) ** 2

    with torch.no_grad():
        codes     = vq_model.encode(images)
        clip_feat = clip_model.encode_text(labels)

    T        = max(1, int(torch.log2(torch.tensor(N, dtype=torch.float)).item()))
    schedule = step_schedule(T).tolist()
    step_idx = torch.randint(0, T + 1, (B,), device=device)

    # --- Single train step ----------------------------------------------------
    print("Running train_step ...")
    ce, bce, acc = trainer.train_step(codes, images, clip_feat, schedule, step_idx)
    print(f"[PASS]  CE={ce:.4f}  BCE={bce:.4f}  Acc={acc:.3f}")


if __name__ == "__main__":
    main()
