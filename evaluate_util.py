"""
FID (Frechet Inception Distance) evaluation.

Requires:  pip install torchmetrics[image]

FID measures the distance between the Inception-v3 feature distributions of
real and generated images.  Lower is better; 0 means identical distributions.

Images fed in must be ImageNet-normalized tensors (B, 3, H, W) — the same
format produced by the DataLoader in data.py.  They are denormalized to
[0, 1] before being passed to the Inception model.
"""

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

from utils import denormalize


def _to_uint8(imgs: Tensor) -> Tensor:
    """Denormalize ImageNet tensors and convert to uint8 (B, 3, H, W) in [0, 255]."""
    return (denormalize(imgs) * 255).to(torch.uint8)


def compute_fid(
    real_imgs: Tensor,
    fake_imgs: Tensor,
    feature: int = 2048,
    device: str = "cpu",
) -> float:
    """Compute FID between two batches of ImageNet-normalized image tensors.

    Args:
        real_imgs: (B, 3, H, W) normalized tensor — real images from the dataset
        fake_imgs: (B, 3, H, W) normalized tensor — generated / reconstructed images
        feature:   Inception feature layer dimension (64 | 192 | 768 | 2048)
        device:    device to run Inception on

    Returns:
        FID score as a Python float (lower is better)

    Note:
        FID estimates are unreliable for very small batches.  Use at least
        2 048 images per split for a stable score.
    """
    fid = FrechetInceptionDistance(feature=feature, normalize=False).to(device)

    fid.update(_to_uint8(real_imgs).to(device), real=True)
    fid.update(_to_uint8(fake_imgs).to(device), real=False)

    return fid.compute().item()


class FIDAccumulator:
    """Streaming FID for images already in [0, 1] (e.g. VQ-GAN decoded outputs).

    Accepts batches one at a time so the full image set never has to sit in
    memory at once.  Call update() per batch, then compute() once at the end.
    """

    def __init__(self, feature: int = 2048, device: str = "cpu"):
        self.fid    = FrechetInceptionDistance(feature=feature, normalize=True).to(device)
        self.device = device

    def update(self, real_imgs: Tensor, fake_imgs: Tensor) -> None:
        """real_imgs / fake_imgs: (B, 3, H, W) float in [0, 1]."""
        self.fid.update(real_imgs.clamp(0, 1).to(self.device), real=True)
        self.fid.update(fake_imgs.clamp(0, 1).to(self.device), real=False)

    def compute(self) -> float:
        return self.fid.compute().item()


def compute_fid_from_loader(
    real_loader: DataLoader,
    fake_loader: DataLoader,
    feature: int = 2048,
    device: str = "cpu",
    max_batches: int = None,
) -> float:
    """Compute FID by streaming batches from two DataLoaders.

    Args:
        real_loader:  DataLoader yielding (imgs, labels) of real images
        fake_loader:  DataLoader yielding (imgs, labels) of generated images
        feature:      Inception feature layer dimension
        device:       device to run Inception on
        max_batches:  stop after this many batches (None = use all)

    Returns:
        FID score as a Python float
    """
    fid = FrechetInceptionDistance(feature=feature, normalize=False).to(device)

    for i, ((real_imgs, _), (fake_imgs, _)) in enumerate(
        zip(real_loader, fake_loader)
    ):
        fid.update(_to_uint8(real_imgs).to(device), real=True)
        fid.update(_to_uint8(fake_imgs).to(device), real=False)
        if max_batches is not None and i + 1 >= max_batches:
            break

    return fid.compute().item()
