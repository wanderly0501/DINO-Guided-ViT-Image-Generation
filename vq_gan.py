"""
MaskGIT VQ-GAN (PretrainedTokenizer) encode/decode example.

Model:  maskgit-vqgan-imagenet-f16-256.bin  (fun-research/TiTok)
        f=16, codebook 1024 entries x 256-dim
        256x256 input -> 16x16 = 256 tokens per image

Requires:
  - 1d-tokenizer cloned into project_vs/1d-tokenizer/
      git clone https://github.com/bytedance/1d-tokenizer.git
  - pip install omegaconf huggingface_hub einops
"""

import sys
import os
import torch
import matplotlib.pyplot as plt

# Add the cloned repo and project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "1d-tokenizer"))
sys.path.insert(0, os.path.dirname(__file__))

from modeling.titok import PretrainedTokenizer
from data import get_hf_dataloaders
from utils import denormalize, mask_to_image

SEED = 263
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Load model  (auto-loads weights, sets eval, freezes params)
# ---------------------------------------------------------------------------

CKPT_PATH = os.path.join(os.path.dirname(__file__), "..", "ckpts", "maskgit-vqgan-imagenet-f16-256.bin")
model = PretrainedTokenizer(CKPT_PATH)
"""
# ---------------------------------------------------------------------------
# Sanity check: random batch  (B, 3, 256, 256) in [0, 1]
# ---------------------------------------------------------------------------

B = 4
data = torch.rand(B, 3, 256, 256)
tokens_batch = model.encode(data)               # (4, 256)
recon_batch  = model.decode(tokens_batch)       # (4, 3, 256, 256)
print(f"batch input:  {tuple(data.shape)}")
print(f"batch tokens: {tuple(tokens_batch.shape)}")
print(f"batch recon:  {tuple(recon_batch.shape)}")

H_tok, W_tok = 256 // 16, 256 // 16
assert tokens_batch.shape == (B, H_tok * W_tok)
assert recon_batch.shape  == data.shape
print("Batch shape assertions passed.")
"""
# ---------------------------------------------------------------------------
# Real val image: encode -> print tokens -> decode -> display
# ---------------------------------------------------------------------------

_, val_loader, _ = get_hf_dataloaders(batch_size=1, num_workers=0, pin_memory=False)
img_norm, label = next(iter(val_loader))        # (1, 3, 256, 256), (1,)
img_norm, label = img_norm[0], label[0].item()

print(f"\nclass label: {label}")

# denormalise ImageNet -> [0, 1] for the VQ-GAN, add batch dim
img_01    = denormalize(img_norm)               # (3, 256, 256) in [0, 1]
img_batch = img_01.unsqueeze(0)                 # (1, 3, 256, 256)

# encode
tokens = model.encode(img_batch)               # (1, 256)
print(f"tokens shape: {tuple(tokens.shape)}")
print(f"tokens:\n{tokens}")

# decode
recon = model.decode(tokens).squeeze(0)        # (3, 256, 256) in [0, 1]

# half-masked decode: zero out the second half of tokens
N = tokens.shape[1]                              # 256 for 16x16 grid
tokens_half = tokens.clone()
tokens_half[:, N // 2 :] = 0                    # mask second half with token 0
recon_half = model.decode(tokens_half).squeeze(0)  # (3, 256, 256) in [0, 1]

# heatmap: green = revealed (first half), gold = masked (second half)
key_mask   = torch.zeros(N, dtype=torch.bool)
key_mask[:N // 2] = True                        # first half revealed
query_mask = ~key_mask                          # second half masked
heatmap = mask_to_image(key_mask, query_mask)   # (3, grid, grid)

# display: original | reconstructed | half-masked | mask heatmap
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
panels = [
    (img_01,    "Original"),
    (recon,     "Reconstructed"),
    (recon_half, "Half codes masked"),
    (heatmap,   "Mask\ngreen=revealed  gold=masked"),
]
for ax, (img, title) in zip(axes, panels):
    ax.imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
    ax.set_title(title, fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.show()

