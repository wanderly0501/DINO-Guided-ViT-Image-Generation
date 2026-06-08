import torch


def load_dino_model(arch: str = "vits8") -> torch.nn.Module:
    """Load a pretrained DINO ViT model via torch.hub.

    arch options with 8x8 patches: 'vits8', 'vitb8'
    """
    model = torch.hub.load("facebookresearch/dino:main", f"dino_{arch}")
    model.eval()
    return model


def get_dino_attention(
    img_tensor: torch.Tensor,
    dino_model: torch.nn.Module,
    merge_heads: bool = True
) -> torch.Tensor:
    """Extract per-head CLS attention maps from the last DINO self-attention layer.

    Args:
        img_tensor: (B, C, H, W) image tensor
        dino_model: DINO model with get_last_selfattention(), loaded via load_dino_model()

    Returns:
        attn: (nh, w_feat, h_feat) — one spatial attention map per head
    """
    b, _, w, h = img_tensor.shape   # _ = colour channels, not attention heads

    with torch.no_grad():
        attn = dino_model.get_last_selfattention(img_tensor)[:, :, 0, 1:]
        # attn: (B, nh, N)  — CLS-to-patch attention per head

    w_feat, h_feat = w // 8, h // 8   # 32x32 for 256x256 input
    nh = attn.shape[1]
    attn = attn.reshape(b, nh, w_feat, h_feat)  # (B, nh, 32, 32)

    # Sum-pool 32x32 -> 16x16 to align with VQ-GAN's 16x16 token grid
    attn = attn.reshape(b, nh, 16, 2, 16, 2).sum(dim=(3, 5))  # (B, nh, 16, 16)
    attn = attn.reshape(b, nh, 16 * 16)                        # (B, nh, 256)

    if merge_heads:
        attn = attn.mean(dim=1)   # (B, 256)  — average across heads
    return attn


def get_patch_sorted_index(
    img_tensor: torch.Tensor,
    dino_model: torch.nn.Module,
) -> torch.Tensor:
    """Return patch indices sorted by DINO attention (most salient first).

    Returns:
        sorted_idx: (B, N) — patch indices in descending attention order
    """
    attn = get_dino_attention(img_tensor, dino_model, merge_heads=True)  # (B, N)
    sorted_patch_index = torch.argsort(attn, dim=-1, descending=True)    # (B, N)
    return sorted_patch_index