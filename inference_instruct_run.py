"""
Text-conditioned inference: generates one image per free-form text prompt using
raw CLIP text embeddings — no ImageNet class indices required.

Usage:
  python inference_instruct_run.py
  python inference_instruct_run.py --checkpoint checkpoints/ckpt_epoch0002_step000500.pt
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
from utils import visualize_steps, reveal_order_to_image
from main import VQGANWrapper

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "1d-tokenizer"))
from modeling.titok import PretrainedTokenizer


CLASSES = [
    "Siamese cat",
    "daisy",
    "pelican",
    "coyote",
    "garter snake",
    "hip",
    "jaguar",
    "viaduct",
    "hummingbird",
    "espresso",
]


def find_latest_checkpoint(save_dir: str) -> str:
    pattern = os.path.join(save_dir, "ckpt_epoch*_step*.pt")
    paths   = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {save_dir!r}")
    return max(paths)


def parse_args():
    parser = argparse.ArgumentParser(description="Text-conditioned MaskGIT inference")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to MaskedViT checkpoint (.pt). "
                             "Defaults to the latest in cfg.train.save_dir.")
    parser.add_argument("--ckpt_path", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "ckpts",
                                             "maskgit-vqgan-imagenet-f16-256.bin"),
                        help="Path to VQ-GAN tokenizer weights.")
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

    print("Loading CLIP ...")
    clip_raw, _ = clip.load("ViT-B/32", device=device)
    clip_raw.eval()

    checkpoint_path = args.checkpoint or find_latest_checkpoint(cfg.train.save_dir)
    print(f"Loading MaskedViT from: {checkpoint_path}")
    model = MaskedViT(cfg.model).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  epoch={ckpt.get('epoch', '?')}  step={ckpt.get('step', '?')}")

    # --- CLIP text embeddings (one per prompt) --------------------------------
    prompts = clip.tokenize([f"a photo of a {c}" for c in CLASSES]).to(device)
    with torch.no_grad():
        clip_feat = clip_raw.encode_text(prompts).float()   # (len(CLASSES), clip_dim)

    # --- Generate + save ------------------------------------------------------
    N = (cfg.model.img_size // 16) ** 2
    T = max(1, int(torch.log2(torch.tensor(N, dtype=torch.float)).item()))
    save_dir = os.path.join(cfg.inference.save_dir, "instruct")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Generating {len(CLASSES)} images -> {save_dir}/")

    for i, class_name in enumerate(CLASSES):
        slug        = class_name.replace(" ", "_")
        clip_feat_i = clip_feat[i : i + 1]   # (1, clip_dim)

        gen, _, rev_step = generate(
            model, vq_model, clip_feat_i,
            N=N, T=T, temperature=cfg.inference.temperature,
            device=str(device),
        )

        heatmap   = reveal_order_to_image(rev_step.squeeze(0), T)
        save_path = os.path.join(save_dir, f"{i:02d}_{slug}.png")
        visualize_steps(
            [
                (gen.squeeze(0).cpu(), f"generated\n{class_name}"),
                (heatmap,              "generation order\npurple→early  yellow→late"),
            ],
            title=class_name,
            save_path=save_path,
        )
        print(f"  [{i:02d}] {class_name} -> {save_path}")

    print(f"\nDone. {len(CLASSES)} images saved to {save_dir}/")


if __name__ == "__main__":
    main()
