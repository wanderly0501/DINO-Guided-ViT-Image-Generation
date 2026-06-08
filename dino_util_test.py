"""
Test dino_util.py:
  - Load one random val image.
  - Plot per-head DINO attention maps.
  - Print get_patch_sorted_index result.

Run with:  python dino_util_test.py
"""

import random
import torch
import matplotlib.pyplot as plt

from data import get_dataset
from utils import denormalize
from dino_util import load_dino_model, get_dino_attention, get_patch_sorted_index


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

SEED = 142
random.seed(SEED)
torch.manual_seed(SEED)

print("Loading DINO model (vits8) ...")
dino_model = load_dino_model(arch="vits8")

print("Loading val dataset ...")
val_dataset = get_dataset("val")
idx         = random.randint(0, len(val_dataset) - 1)
img_norm, label = val_dataset[idx]      # (3, 256, 256) ImageNet-normalised

print(f"Image index: {idx}  |  class label: {label}")

# DINO expects ImageNet-normalised input; keep img_norm as-is.
# Add batch dim: (1, 3, 256, 256)
img_batch = img_norm.unsqueeze(0)

# ---------------------------------------------------------------------------
# Attention maps (per head)
# ---------------------------------------------------------------------------

print("\n--- get_dino_attention (per head) ---")
attn_per_head = get_dino_attention(img_batch, dino_model, merge_heads=False)
# attn_per_head: (1, nh, N)
nh = attn_per_head.shape[1]
N  = attn_per_head.shape[2]
H_feat = W_feat = int(N ** 0.5)
print(f"attn shape: {tuple(attn_per_head.shape)}  "
      f"(batch=1, heads={nh}, patches={N} = {H_feat}x{W_feat})")

# Show original image + one attention map per head
n_cols  = nh + 1
_, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2.5, 3))

# Original image
axes[0].imshow(denormalize(img_norm).permute(1, 2, 0).numpy())
axes[0].set_title("Original")
axes[0].axis("off")

# Per-head attention maps
for h in range(nh):
    attn_map = attn_per_head[0, h].reshape(H_feat, W_feat).cpu().numpy()
    axes[h + 1].imshow(attn_map, cmap="inferno")
    axes[h + 1].set_title(f"Head {h}")
    axes[h + 1].axis("off")

plt.suptitle("DINO CLS attention maps (per head)", fontsize=10)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Merged attention map
# ---------------------------------------------------------------------------

print("\n--- get_dino_attention (merge_heads=True) ---")
attn_merged = get_dino_attention(img_batch, dino_model, merge_heads=True)
# attn_merged: (1, N)
print(f"merged attn shape: {tuple(attn_merged.shape)}")

attn_map = attn_merged[0].reshape(H_feat, W_feat).cpu().numpy()
_, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(denormalize(img_norm).permute(1, 2, 0).numpy())
axes[0].set_title("Original")
axes[0].axis("off")
axes[1].imshow(attn_map, cmap="inferno")
axes[1].set_title("Merged attention")
axes[1].axis("off")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# get_patch_sorted_index
# ---------------------------------------------------------------------------

print("\n--- get_patch_sorted_index ---")
sorted_idx = get_patch_sorted_index(img_batch, dino_model)  # (1, N)
print(f"sorted_idx shape: {tuple(sorted_idx.shape)}")
print(f"Top-10 most salient patches: {sorted_idx[0, :10].tolist()}")
print(f"Bottom-10 least salient:     {sorted_idx[0, -10:].tolist()}")

# Visualise sorted order as a heatmap (rank of each patch)
rank_map = torch.zeros(N)
rank_map[sorted_idx[0]] = torch.arange(N, dtype=torch.float)
rank_map = rank_map.reshape(H_feat, W_feat).numpy()

_, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(denormalize(img_norm).permute(1, 2, 0).numpy())
axes[0].set_title("Original")
axes[0].axis("off")
axes[1].imshow(rank_map, cmap="viridis")
axes[1].set_title("Patch reveal order\n(dark = revealed first)")
axes[1].axis("off")
plt.tight_layout()
plt.show()

print("\nAll tests passed.")
