"""
ImageNet (ILSVRC2012) data loading utilities.

Local mode (default):
  Expects the dataset pre-extracted under `dataset_dir` (default: ./dataset):
    dataset/
      ILSVRC2012_devkit_t12/data/meta.mat
      ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt
      train/  <n01440764>/  *.JPEG  ...
      val/    *.JPEG  (flat — prepare_val() organises into class folders on first use)

HuggingFace mode (use get_hf_dataloaders):
  Requires accepting the license at https://huggingface.co/datasets/ILSVRC/imagenet-1k
  then: pip install datasets

Requires: torch, torchvision, scipy (for devkit meta.mat parsing).
"""

import functools
import os
import shutil
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Polyfill for Pillow < 9.2.0: PIL.Image.ExifTags and ExifTags.Base were added
# in 9.2.0 but the `datasets` library requires them during image decoding.
import PIL.Image
import PIL.ExifTags
if not hasattr(PIL.Image, "ExifTags"):
    PIL.Image.ExifTags = PIL.ExifTags
if not hasattr(PIL.ExifTags, "Base"):
    from enum import IntEnum
    class _ExifBase(IntEnum):
        Orientation = 274
    PIL.ExifTags.Base = _ExifBase
    PIL.Image.ExifTags = PIL.ExifTags

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

Split = Literal["train", "val"]


def get_transforms(split: Split, img_size: int = 256) -> transforms.Compose:
    """ImageNet transforms: resize shorter edge to img_size, then crop to img_size×img_size.

    Train: resize → random crop + horizontal flip + colour jitter → normalise.
    Val:   resize → centre crop → normalise.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def prepare_val(dataset_dir: str = "dataset") -> None:
    """Organise flat val images into per-class subdirectories using devkit labels.

    Runs once: if class subdirectories already exist the function returns immediately.
    Requires scipy.
    """
    import scipy.io

    val_dir = os.path.join(os.path.abspath(dataset_dir), "val")
    # Already organised if any subdirectory exists
    if any(os.path.isdir(os.path.join(val_dir, e)) for e in os.listdir(val_dir)):
        return

    devkit_dir = os.path.join(os.path.abspath(dataset_dir), "ILSVRC2012_devkit_t12", "data")

    # Map 1-indexed ILSVRC2012_ID -> WNID
    meta = scipy.io.loadmat(os.path.join(devkit_dir, "meta.mat"), squeeze_me=True)
    id_to_wnid = {int(s["ILSVRC2012_ID"]): str(s["WNID"]) for s in meta["synsets"]}

    # Ground truth: 50 000 lines, each a 1-indexed class ID
    gt_path = os.path.join(devkit_dir, "ILSVRC2012_validation_ground_truth.txt")
    with open(gt_path) as f:
        class_ids = [int(line.strip()) for line in f]

    val_images = sorted(fn for fn in os.listdir(val_dir) if fn.endswith(".JPEG"))

    # Create class folders
    for wnid in {id_to_wnid[cid] for cid in class_ids}:
        os.makedirs(os.path.join(val_dir, wnid), exist_ok=True)

    # Move each image into its class folder
    for img_name, class_id in zip(val_images, class_ids):
        wnid = id_to_wnid[class_id]
        shutil.move(
            os.path.join(val_dir, img_name),
            os.path.join(val_dir, wnid, img_name),
        )
    print(f"Val images organised into {len(set(class_ids))} class folders.")


def get_dataset(
    split: Split,
    dataset_dir: str = "dataset",
    img_size: int = 256,
) -> ImageFolder:
    """Return an ImageNet dataset for the given split using ImageFolder.

    For val, calls prepare_val() on first use to organise flat images into
    class subdirectories.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    if split == "val":
        prepare_val(dataset_dir)
    return ImageFolder(
        root=os.path.join(dataset_dir, split),
        transform=get_transforms(split, img_size),
    )


def get_dataloader(
    split: Split,
    dataset_dir: str = "dataset",
    batch_size: int = 256,
    num_workers: int = 8,
    img_size: int = 256,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Return a DataLoader for the given ImageNet split."""
    dataset = get_dataset(split, dataset_dir, img_size)
    shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def get_dataloaders(
    dataset_dir: str = "dataset",
    batch_size: int = 256,
    num_workers: int = 8,
    img_size: int = 256,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for ImageNet.

    Example::

        train_loader, val_loader = get_dataloaders(batch_size=128, num_workers=4)
        for images, labels in train_loader:
            ...
    """
    train_loader = get_dataloader(
        "train", dataset_dir, batch_size, num_workers, img_size,
        pin_memory, drop_last=True,
    )
    val_loader = get_dataloader(
        "val", dataset_dir, batch_size, num_workers, img_size,
        pin_memory, drop_last=False,
    )
    return train_loader, val_loader


@functools.lru_cache(maxsize=1)
def load_class_names(dataset_dir: str = "dataset") -> dict[str, str]:
    """Return a dict mapping WNID to human-readable class name.

    Parses ILSVRC2012_devkit_t12/data/meta.mat. Requires scipy.
    Result is cached so the .mat file is only read once per dataset_dir.
    """
    import scipy.io
    meta_path = os.path.join(
        os.path.abspath(dataset_dir),
        "ILSVRC2012_devkit_t12", "data", "meta.mat",
    )
    meta = scipy.io.loadmat(meta_path, squeeze_me=True)
    synsets = meta["synsets"]
    return {str(s["WNID"]): str(s["words"]) for s in synsets}


def get_class_name(wnid: str, dataset_dir: str = "dataset") -> str:
    """Return the human-readable name for a WNID, e.g. 'n01440764' -> 'tench, Tinca tinca'."""
    return load_class_names(dataset_dir).get(wnid, wnid)


# ---------------------------------------------------------------------------
# Canonical ImageNet class names (int index → readable name)
# ---------------------------------------------------------------------------

def get_imagenet_class_names() -> list[str]:
    """Return the 1000 ImageNet class names ordered by label index (0=tench, 1=goldfish, ...).

    Reads from 1d-tokenizer/imagenet_classes.py so no local devkit is needed.
    Works for both local ImageFolder and HuggingFace datasets since both use
    the same standard integer ordering.
    """
    import sys
    tokenizer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "1d-tokenizer")
    if tokenizer_dir not in sys.path:
        sys.path.append(tokenizer_dir)
    from imagenet_classes import imagenet_idx2classname
    return [imagenet_idx2classname[i] for i in range(len(imagenet_idx2classname))]


# ---------------------------------------------------------------------------
# HuggingFace loader
# ---------------------------------------------------------------------------

class HFImageNetDataset(Dataset):
    """Wraps a HuggingFace ILSVRC/imagenet-1k split to match ImageFolder's (image, label) API.

    Requires accepting the license at https://huggingface.co/datasets/ILSVRC/imagenet-1k
    and installing the `datasets` package.
    """

    def __init__(self, hf_split, transform=None):
        self.data      = hf_split
        self.transform = transform
        self.classes   = get_imagenet_class_names()  # readable names indexed by label int

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item  = self.data[idx]
        image = item["image"].convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, item["label"]


def get_hf_dataloaders(
    batch_size:  int  = 256,
    num_workers: int  = 8,
    img_size:    int  = 256,
    pin_memory:  bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) for ImageNet from HuggingFace.

    Requires:
      pip install datasets
      License accepted at https://huggingface.co/datasets/ILSVRC/imagenet-1k
    """
    from datasets import load_dataset
    hf_token = os.environ.get("HF_TOKEN")
    ds = load_dataset("ILSVRC/imagenet-1k", token=hf_token)

    train_dataset = HFImageNetDataset(ds["train"],      get_transforms("train", img_size))
    val_dataset   = HFImageNetDataset(ds["validation"], get_transforms("val",   img_size))
    test_dataset  = HFImageNetDataset(ds["test"],       get_transforms("val",   img_size))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    return train_loader, val_loader, test_loader
