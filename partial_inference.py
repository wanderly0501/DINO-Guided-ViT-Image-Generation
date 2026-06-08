"""
Partial inference: encode real images, reveal context up to a given step,
then let the model complete the rest.

Usage:
  python partial_inference.py
  python partial_inference.py --checkpoint checkpoints/ckpt_epoch0002_step000500.pt
  python partial_inference.py --start_step 5
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
from inference import generate_from_partial
from data import get_hf_dataloaders, get_imagenet_class_names
from utils import visualize_steps, reveal_order_to_image, step_schedule, make_masks
from dino_util import get_patch_sorted_index
from main import VQGANWrapper, CLIPWrapper

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "1d-tokenizer"))
from modeling.titok import PretrainedTokenizer


def find_latest_checkpoint(save_dir: str) -> str:
    pattern = os.path.join(save_dir, "ckpt_epoch*_step*.pt")
    paths   = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {save_dir!r}")
    return max(paths)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--ckpt_path",  type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "ckpts", "maskgit-vqgan-imagenet-f16-256.bin"))
    parser.add_argument("--start_step", type=int, default=8,
                        help="Step index at which context ends and generation begins.")
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.train.device = str(device)
    cfg.model.disable_mask_in_attention = True
    print(f"Device: {device}  |  start_step={args.start_step}")

    # --- Models ---------------------------------------------------------------
    print("Loading VQ-GAN ...")
    vq_model = VQGANWrapper(PretrainedTokenizer(args.ckpt_path)).to(device)

    print("Loading CLIP ...")
    clip_raw, _ = clip.load("ViT-B/32", device=device)
    clip_raw.eval()
    class_names = get_imagenet_class_names()
    clip_model  = CLIPWrapper(clip_raw, class_names, device)

    print("Loading DINO ...")
    from dino_util import load_dino_model
    dino_model = load_dino_model(arch=cfg.dino.arch).to(device).eval()

    checkpoint_path = args.checkpoint or find_latest_checkpoint(cfg.train.save_dir)
    print(f"Loading MaskedViT from: {checkpoint_path}")
    model = MaskedViT(cfg.model).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  epoch={ckpt.get('epoch', '?')}  step={ckpt.get('step', '?')}")

    # --- Val data -------------------------------------------------------------
    print("Loading val dataset ...")
    _, val_loader, _ = get_hf_dataloaders(
        batch_size=cfg.inference.n_samples,
        num_workers=0,
        pin_memory=False,
        img_size=cfg.model.img_size,
    )
    images, labels = next(iter(val_loader))
    images = images.to(device)
    labels = labels.to(device)
    B      = images.shape[0]

    # --- Encode & condition ---------------------------------------------------
    with torch.no_grad():
        codes     = vq_model.encode(images)                        # (B, N)
        clip_feat = clip_model.encode_text(labels)                 # (B, clip_dim)
        sorted_idx = get_patch_sorted_index(images, dino_model).to(device)  # (B, N)

    N        = codes.shape[1]
    T        = max(1, int(torch.log2(torch.tensor(N, dtype=torch.float)).item()))
    schedule = step_schedule(T).tolist()

    # --- Build masks at start_step -------------------------------------------
    start_step = min(args.start_step, T)
    step_idx   = torch.full((B,), start_step, dtype=torch.long, device=device)
    query_mask, key_mask, _ = make_masks(sorted_idx, schedule, step_idx, N, B, device=device)
    # key_mask:   tokens revealed BEFORE start_step  (context)
    # query_mask: tokens to reveal AT start_step     (first predictions)

    # --- Generate -------------------------------------------------------------
    save_dir = os.path.join(cfg.inference.save_dir, f"partial_step{start_step}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Generating {B} images (context = step {start_step}, saving to {save_dir}) ...")

    for i in range(B):
        class_name  = class_names[labels[i].item()].split(",")[0]
        slug        = class_name.replace(" ", "_")
        clip_feat_i = clip_feat[i:i+1]
        codes_i     = codes[i:i+1]
        key_i       = key_mask[i:i+1]
        query_i     = query_mask[i:i+1]

        gen, _, rev_step = generate_from_partial(
            model, vq_model, clip_feat_i,
            context_codes=codes_i,
            key_mask=key_i,
            initial_query_mask=query_i,
            start_step=start_step,
            N=N, T=T,
            temperature=cfg.inference.temperature,
            device=str(device),
            greedy=True,
        )

        # Partial image: only context tokens decoded, rest zeroed
        partial_codes = torch.full_like(codes_i, 0)
        partial_codes[key_i] = codes_i[key_i]
        partial_img = vq_model.decode(partial_codes).squeeze(0).cpu()

        orig_decoded = vq_model.decode(codes_i).squeeze(0).cpu()
        heatmap      = reveal_order_to_image(rev_step.squeeze(0), T)

        save_path = os.path.join(save_dir, f"{i:04d}_{slug}.png")
        visualize_steps(
            [
                (orig_decoded,         f"original (decoded)\n{class_name}"),
                (partial_img,          f"context (step {start_step})\n{int(key_i.sum())} tokens"),
                (gen.squeeze(0).cpu(), f"completed\n{class_name}"),
                (heatmap,              f"generation order\npurple→early  yellow→late"),
            ],
            title=f"{class_name}  —  context up to step {start_step}",
            save_path=save_path,
        )
        print(f"  [{i}] {class_name} -> {save_path}")

    print(f"\nDone. {B} images saved to {save_dir}")


if __name__ == "__main__":
    main()
