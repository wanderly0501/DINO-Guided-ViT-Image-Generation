import torch
import matplotlib.pyplot as plt
from torch.nn.init import trunc_normal_  # re-exported so DINO's internal imports resolve

# ImageNet normalization constants (must match data.py)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(img: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization. Input: (3, H, W), output: (3, H, W) in [0, 1]."""
    mean = _MEAN.to(img.device)
    std  = _STD.to(img.device)
    return (img * std + mean).clamp(0.0, 1.0)


def show_image(
    img: torch.Tensor,
    title: str = "",
    ax: plt.Axes = None,
) -> None:
    """Display a single normalized image tensor with matplotlib.

    Args:
        img:   (3, H, W) float tensor, ImageNet-normalized
        title: optional title shown above the image
        ax:    existing Axes to draw into; creates a new figure if None
    """
    rgb = denormalize(img).permute(1, 2, 0).numpy()  # (H, W, 3)

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(rgb)
    ax.axis("off")
    if title:
        ax.set_title(title)

    if ax is None:
        plt.tight_layout()
        plt.show()


def show_batch(
    imgs: torch.Tensor,
    titles: list[str] = None,
    ncols: int = 8,
) -> None:
    """Display a batch of normalized image tensors in a grid.

    Args:
        imgs:   (B, 3, H, W) float tensor, ImageNet-normalized
        titles: optional list of per-image title strings
        ncols:  number of columns in the grid
    """
    B = imgs.shape[0]
    nrows = (B + ncols - 1) // ncols
    _, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = axes.flatten() if B > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < B:
            title = titles[i] if titles else ""
            show_image(imgs[i], title=title, ax=ax)
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def get_2d_sinusoidal_pos_embed(embed_dim: int, grid_size: int = 16) -> torch.Tensor:
    """2D sinusoidal positional embedding for a grid_size x grid_size patch grid.

    Half of embed_dim encodes row position, the other half encodes column position.
    Follows the standard ViT/MAE sinusoidal formulation.

    Args:
        embed_dim: total embedding dimension (must be even)
        grid_size: number of patches along each spatial axis (default 16 for 256/16)

    Returns:
        pos_embed: (1, grid_size*grid_size, embed_dim)
                   ready to add directly to patch token embeddings
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"

    half_dim = embed_dim // 2                                    # half for rows, half for cols

    # 1-D sinusoidal encoding for a single axis
    freq = torch.arange(half_dim // 2, dtype=torch.float32)
    freq = 1.0 / (10000 ** (2 * freq / half_dim))               # (half_dim/2,)
    pos  = torch.arange(grid_size, dtype=torch.float32)
    ang  = pos.unsqueeze(1) * freq.unsqueeze(0)                  # (grid_size, half_dim/2)
    enc_1d = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1) # (grid_size, half_dim)

    # Expand to 2-D grid
    row_ids = torch.arange(grid_size).repeat_interleave(grid_size)  # (N,) row index per token
    col_ids = torch.arange(grid_size).repeat(grid_size)             # (N,) col index per token

    pos_embed = torch.cat([enc_1d[row_ids], enc_1d[col_ids]], dim=1)  # (N, embed_dim)
    return pos_embed.unsqueeze(0)                                       # (1, N, embed_dim)


def step_schedule(steps: int = 8) -> torch.Tensor:
    """Return a tensor of length steps+1 where index t holds 2^t.

    Args:
        steps: number of steps (resulting tensor has steps+1 elements)

    Returns:
        schedule: (steps+1,) int64 tensor  [2^0, 2^1, ..., 2^steps]

    Example:
        step_schedule(4) -> tensor([ 1,  2,  4,  8, 16])
    """
    t = torch.arange(steps + 1)
    return 2 ** t


def make_masks(
    sorted_idx: torch.Tensor,   # (B, N)  per-image DINO ordering
    schedule:   list,
    step_idx:   torch.Tensor,   # (B,)    random step per image
    N:          int,
    B:          int,
    device:     torch.device = None,
):
    """Build query_mask, key_mask, next_query using each image's own random step.

    For image b at step t = step_idx[b]:
      key_mask[b]:    sorted_idx[b, 0 : schedule[t-1]]            -- context tokens
      query_mask[b]:  sorted_idx[b, schedule[t-1] : schedule[t]]  -- predicted now
      next_query[b]:  sorted_idx[b, schedule[t] : schedule[t+1]]  -- predicted next
      Positions schedule[t+1].. are fully masked (not in any mask).

    Returns:
        query_mask, key_mask, next_query  each (B, N) bool
    """
    if device is None:
        device = sorted_idx.device

    query_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    key_mask   = torch.zeros(B, N, dtype=torch.bool, device=device)
    next_query = torch.zeros(B, N, dtype=torch.bool, device=device)

    for b in range(B):
        t         = int(step_idx[b].item())
        key_end   = schedule[t - 1] if t > 0 else 0
        query_end = schedule[t]
        next_end  = schedule[t + 1] if t + 1 < len(schedule) else N

        key_mask  [b, sorted_idx[b, :key_end]]           = True
        query_mask[b, sorted_idx[b, key_end:query_end]]  = True
        next_query[b, sorted_idx[b, query_end:next_end]] = True

    return query_mask, key_mask, next_query


def reveal_order_to_image(reveal_step: torch.Tensor, T: int) -> torch.Tensor:
    """Return a (3, grid, grid) float tensor coloring each patch by its generation step.

    Colormap: plasma — early steps → dark purple, late steps → bright yellow.
    Unrevealed positions (value -1) are shown as dark gray.
    """
    import matplotlib.cm as cm
    import numpy as np
    N    = reveal_step.shape[0]
    grid = int(N ** 0.5)

    step_np    = reveal_step.cpu().float().numpy()
    unrevealed = step_np < 0
    step_norm  = np.clip(step_np / max(T, 1), 0.0, 1.0)

    colored = cm.plasma(step_norm)[:, :3].astype(np.float32)   # (N, 3)
    colored[unrevealed] = [0.15, 0.15, 0.15]

    return torch.from_numpy(colored).reshape(grid, grid, 3).permute(2, 0, 1)  # (3, grid, grid)


def mask_to_image(
    key_mask:   torch.Tensor,                    # (N,) bool — revealed / context
    query_mask: torch.Tensor,                    # (N,) bool — predicted this step
    next_query: torch.Tensor | None = None,      # (N,) bool — predicted next step
) -> torch.Tensor:
    """Return a (3, grid, grid) float tensor color-coding the mask state.

    Green      = key_mask   (context / already revealed)
    Gold       = query_mask (predicted this step)
    Light-blue = next_query (predicted next step, optional)
    Dark-gray  = fully masked
    """
    N    = key_mask.shape[0]
    grid = int(N ** 0.5)

    colors = torch.tensor([
        [0.20, 0.20, 0.20],   # 0: fully masked — dark gray
        [0.40, 0.70, 1.00],   # 1: next_query   — light blue
        [1.00, 0.80, 0.00],   # 2: query_mask   — gold
        [0.20, 0.80, 0.30],   # 3: key_mask     — green
    ])

    label = torch.zeros(N, dtype=torch.long)
    if next_query is not None:
        label[next_query.cpu()] = 1
    label[query_mask.cpu()] = 2
    label[key_mask.cpu()]   = 3           # key overrides lower priorities

    rgb = colors[label]                               # (N, 3)
    return rgb.reshape(grid, grid, 3).permute(2, 0, 1)  # (3, grid, grid)


def visualize_steps(
    snapshots:  list[tuple[torch.Tensor, str]],
    title:      str  = "MaskGIT generation steps",
    save_path:  str  = None,
) -> None:
    """Display a row of (image_tensor, label) snapshots.

    If save_path is given the figure is saved there instead of displayed.
    """
    import os
    n_cols  = len(snapshots)
    _, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2.5, 3))
    if n_cols == 1:
        axes = [axes]

    for ax, (img, label) in zip(axes, snapshots):
        ax.imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
        ax.set_title(label, fontsize=7)
        ax.axis("off")

    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# ---------------------------------------------------------------------------
# Visualization: training loss curves
# ---------------------------------------------------------------------------

def plot_training_losses(
    ce_losses:  list[float],
    bce_losses: list[float],
    title: str = "Training loss",
) -> None:
    """Plot CE and BCE loss curves recorded during training."""
    steps = range(len(ce_losses))
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(steps, ce_losses, color="steelblue")
    ax1.set_title("Cross-entropy loss (token prediction)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")

    ax2.plot(steps, bce_losses, color="coral")
    ax2.set_title("Binary cross-entropy loss (next-token selection)")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()