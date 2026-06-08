"""
FID evaluation via partial inference at the final unmasking step (T).

For each val image:
  1. Encode with VQ-GAN -> token codes.
  2. Build DINO-sorted context up to step T (first N/2 tokens revealed).
  3. Complete the remaining tokens with the MaskGIT model (greedy).
  4. Decode both the completed and ground-truth tokens with VQ-GAN.
  5. Compare the two sets with FID via evaluate_util.FIDAccumulator.

Images are saved to fid_compute/real/ and fid_compute/fake/.
FID score is written to fid_compute/fid_score.txt.

Usage:
  python evaluate.py
  python evaluate.py --checkpoint checkpoints/ckpt_epoch0004_step000500.pt
  python evaluate.py --n_images 5000 --batch_size 32
"""

import argparse
import glob
import os
import sys

# pyarrow (pulled in by datasets) must be imported before torch on Windows
# to avoid a DLL access-violation crash.
import pyarrow  # noqa: F401

import torch
import torchvision
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="Corrupt EXIF data", category=UserWarning)
import clip

from configuration import Config
from maskgit_model import MaskedViT
from inference import generate_from_partial
from data import get_hf_dataloaders, get_imagenet_class_names
from utils import step_schedule, make_masks
from dino_util import get_patch_sorted_index, load_dino_model
from evaluate_util import FIDAccumulator
from main import VQGANWrapper, CLIPWrapper

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "1d-tokenizer"))
from modeling.titok import PretrainedTokenizer

FID_DIR = os.path.join(os.path.dirname(__file__), "..", "fid_compute")


def find_latest_checkpoint(save_dir: str) -> str:
    pattern = os.path.join(save_dir, "ckpt_epoch*_step*.pt")
    paths   = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {save_dir!r}")
    return max(paths)


def parse_args():
    parser = argparse.ArgumentParser(description="FID evaluation via partial inference")
    parser.add_argument("--checkpoint",  type=str, default=None)
    parser.add_argument("--ckpt_path",   type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "ckpts", "maskgit-vqgan-imagenet-f16-256.bin"))
    parser.add_argument("--n_images",    type=int, default=5000,
                        help="Number of image pairs to generate for FID.")
    parser.add_argument("--batch_size",  type=int, default=32)
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.train.device = str(device)
    cfg.model.disable_mask_in_attention = True

    real_dir = os.path.join(FID_DIR, "real")
    fake_dir = os.path.join(FID_DIR, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Target: {args.n_images} image pairs  |  saving to {FID_DIR}/")

    # --- Data first (pyarrow must initialise before torch.hub on Windows) -----
    print("Loading val dataset ...")
    class_names = get_imagenet_class_names()
    _, val_loader, _ = get_hf_dataloaders(
        batch_size=args.batch_size,
        num_workers=0,
        img_size=cfg.model.img_size,
    )

    # --- Models ---------------------------------------------------------------
    print("Loading VQ-GAN ...")
    vq_model = VQGANWrapper(PretrainedTokenizer(args.ckpt_path)).to(device)

    print("Loading CLIP ...")
    clip_raw, _ = clip.load("ViT-B/32", device=device)
    clip_raw.eval()
    clip_model = CLIPWrapper(clip_raw, class_names, device)

    print("Loading DINO ...")
    dino_model = load_dino_model(arch=cfg.dino.arch).to(device).eval()

    checkpoint_path = args.checkpoint or find_latest_checkpoint(cfg.train.save_dir)
    print(f"Loading MaskedViT from: {checkpoint_path}")
    model = MaskedViT(cfg.model).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  epoch={ckpt.get('epoch', '?')}  step={ckpt.get('step', '?')}")

    # --- Schedule / step setup ------------------------------------------------
    N          = (cfg.model.img_size // 16) ** 2
    T          = max(1, int(torch.log2(torch.tensor(N, dtype=torch.float)).item()))
    schedule   = step_schedule(T).tolist()
    start_step = T   # final unmasking step

    n_context = schedule[T - 1]          # tokens already revealed
    n_query   = N - n_context            # tokens to complete
    print(f"Step T={T}: {n_context} context tokens, {n_query} tokens to predict")

    # --- Streaming FID --------------------------------------------------------
    fid_acc = FIDAccumulator(feature=2048, device=str(device))
    total   = 0
    img_idx = 0

    for images, labels in val_loader:
        if total >= args.n_images:
            break

        images = images.to(device)
        labels = labels.to(device)
        B = images.shape[0]

        with torch.no_grad():
            codes      = vq_model.encode(images)                              # (B, N)
            clip_feat  = clip_model.encode_text(labels)                       # (B, clip_dim)
            sorted_idx = get_patch_sorted_index(images, dino_model).to(device)  # (B, N)

        step_idx = torch.full((B,), start_step, dtype=torch.long, device=device)
        query_mask, key_mask, _ = make_masks(
            sorted_idx, schedule, step_idx, N, B, device=device
        )

        # Complete the image from context — returns (B, 3, H, W) in [0, 1]
        gen_imgs, _, _ = generate_from_partial(
            model, vq_model, clip_feat,
            context_codes=codes,
            key_mask=key_mask,
            initial_query_mask=query_mask,
            start_step=start_step,
            N=N, T=T,
            device=str(device),
            greedy=True,
        )

        # Ground-truth decoded images — (B, 3, H, W) in [0, 1]
        with torch.no_grad():
            real_imgs = vq_model.decode(codes)

        # Stream into FID accumulator
        fid_acc.update(real_imgs, gen_imgs)

        # Save to disk (only up to n_images)
        n_save = min(B, args.n_images - total)
        for j in range(n_save):
            torchvision.utils.save_image(
                real_imgs[j].cpu(), os.path.join(real_dir, f"{img_idx:05d}.png")
            )
            torchvision.utils.save_image(
                gen_imgs[j].cpu(),  os.path.join(fake_dir, f"{img_idx:05d}.png")
            )
            img_idx += 1

        total += B
        print(f"  {min(total, args.n_images)}/{args.n_images}", end="\r")

    print(f"\nProcessed {img_idx} image pairs.")

    # --- Compute and report FID -----------------------------------------------
    print("Computing FID ...")
    fid_score = fid_acc.compute()
    print(f"\nFID score: {fid_score:.4f}")
    print(f"  real -> {real_dir}/")
    print(f"  fake -> {fake_dir}/")

    score_path = os.path.join(FID_DIR, "fid_score.txt")
    with open(score_path, "w") as f:
        f.write(f"checkpoint: {checkpoint_path}\n")
        f.write(f"n_images: {img_idx}\n")
        f.write(f"start_step (T): {start_step}\n")
        f.write(f"context_tokens: {n_context}\n")
        f.write(f"query_tokens: {n_query}\n")
        f.write(f"FID: {fid_score:.4f}\n")
    print(f"  score saved to {score_path}")


if __name__ == "__main__":
    main()
