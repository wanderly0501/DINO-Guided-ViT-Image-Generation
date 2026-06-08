"""
Inference run: auto-loads the latest checkpoint from cfg.train.save_dir,
generates class-conditional images from test data, and saves to cfg.inference.save_dir.

Usage:
  python inference_run.py
  python inference_run.py --checkpoint checkpoints/ckpt_epoch0002_step000500.pt  # override
"""

import argparse
import glob
import os
import sys
import torch
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="Corrupt EXIF data", category=UserWarning)
import clip

from configuration import Config
from maskgit_model import MaskedViT
from inference import generate
from data import get_hf_dataloaders, get_imagenet_class_names
from utils import denormalize, visualize_steps, reveal_order_to_image
from main import VQGANWrapper, CLIPWrapper

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "1d-tokenizer"))
from modeling.titok import PretrainedTokenizer


def find_latest_checkpoint(save_dir: str) -> str:
    """Return the path of the most recent checkpoint in save_dir (highest epoch then step)."""
    pattern = os.path.join(save_dir, "ckpt_epoch*_step*.pt")
    paths   = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {save_dir!r}")
    # Filenames sort lexicographically by epoch then step due to zero-padding.
    return max(paths)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to MaskedViT checkpoint (.pt). "
                             "Defaults to the latest in cfg.train.save_dir.")
    parser.add_argument("--ckpt_path", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "ckpts", "maskgit-vqgan-imagenet-f16-256.bin"),
                        help="Path to VQ-GAN tokenizer weights")
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.train.device = str(device)
    print(f"Device: {device}")
    print(f"Inference config: n_samples={cfg.inference.n_samples}  "
          f"T={cfg.inference.T}  temperature={cfg.inference.temperature}  "
          f"save_dir={cfg.inference.save_dir}")

    # --- Models ---------------------------------------------------------------
    print("Loading VQ-GAN ...")
    vq_model = VQGANWrapper(PretrainedTokenizer(args.ckpt_path)).to(device)

    print("Loading CLIP ...")
    clip_raw, _ = clip.load("ViT-B/32", device=device)
    clip_raw.eval()
    class_names = get_imagenet_class_names()
    clip_model  = CLIPWrapper(clip_raw, class_names, device)

    checkpoint_path = args.checkpoint or find_latest_checkpoint(cfg.train.save_dir)
    print(f"Loading MaskedViT from: {checkpoint_path}")
    model = MaskedViT(cfg.model).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  epoch={ckpt.get('epoch', '?')}  step={ckpt.get('step', '?')}")

    # --- Test dataset ---------------------------------------------------------
    print("Loading val dataset ...")
    _, val_loader, _ = get_hf_dataloaders(
        batch_size=cfg.inference.n_samples,
        num_workers=0,
        pin_memory=False,
        img_size=cfg.model.img_size,
    )
    
    val_iter             = iter(val_loader)
    test_images, labels  = next(val_iter)
    test_images2, labels2 = next(val_iter)
    test_images3, labels3 = next(val_iter)
    test_images  = torch.cat([test_images,  test_images2, test_images3], dim=0).to(device)
    labels       = torch.cat([labels,       labels2,      labels3],      dim=0).to(device)

    # --- CLIP class conditioning ----------------------------------------------
    clip_feat = clip_model.encode_text(labels)   # (B, clip_dim)

    # --- Generate + Save ------------------------------------------------------
    N = (cfg.model.img_size // 16) ** 2
    T = max(1, int(torch.log2(torch.tensor(N, dtype=torch.float)).item()))  # match training schedule
    os.makedirs(cfg.inference.save_dir, exist_ok=True)
    print(f"Generating {len(labels)} images ...")

    generated_list = []
    for i, (orig, label) in enumerate(zip(test_images, labels.tolist())):
        class_name   = class_names[label].split(",")[0]
        slug         = class_name.replace(" ", "_")
        clip_feat_i  = clip_feat[i : i + 1]           # (1, clip_dim)

        gen, gen_codes, rev_step = generate(
            model, vq_model, clip_feat_i,
            N=N, T=T, temperature=cfg.inference.temperature,
            device=str(device),
        )   # (1, 3, H, W) in [0, 1]
        generated_list.append(gen.squeeze(0))

        orig_01   = denormalize(orig).cpu()
        heatmap   = reveal_order_to_image(rev_step.squeeze(0), T)
        save_path = os.path.join(cfg.inference.save_dir, f"{i:04d}_{slug}.png")
        visualize_steps(
            [
                (orig_01,              f"original\n{class_name}"),
                (gen.squeeze(0).cpu(), f"generated\n{class_name}"),
                (heatmap,              f"generation order\npurple→early  yellow→late"),
            ],
            title=class_name,
            save_path=save_path,
        )
        print(f"  [{i}] {class_name} -> {save_path}")

    print(f"\nDone. {len(generated_list)} images saved to {cfg.inference.save_dir}")


if __name__ == "__main__":
    main()
