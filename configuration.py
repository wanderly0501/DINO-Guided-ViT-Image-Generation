import os as _os
from dataclasses import dataclass, field

_PROJECT_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))


@dataclass
class DataConfig:
    dataset_dir: str   = _os.path.join(_PROJECT_ROOT, "dataset")
    img_size:    int   = 256
    batch_size:  int   =  256
    num_workers: int   = 4
    pin_memory:  bool  = True

    # ColorJitter (train augmentation)
    jitter_brightness: float = 0.1
    jitter_contrast:   float = 0.1
    jitter_saturation: float = 0.1
    jitter_hue:        float = 0.1


@dataclass
class ModelConfig:
    img_size:    int   = 256
    patch_size:  int   = 8
    embed_dim:   int   = 768
    depth:       int   = 24
    num_heads:   int   = 12
    mlp_ratio:     float = 4.0
    attn_drop:     float = 0.1
    proj_drop:     float = 0.1
    codebook_size: int   = 1024  # VQ-GAN codebook entries (MaskGIT-VQGAN default)
    clip_dim:      int   = 512   # CLIP image embedding dim (ViT-B/32 = 512)
    max_steps:     int   = 8    # upper bound for T (number of unmasking steps)
    disable_mask_in_attention: bool  = False


@dataclass
class DinoConfig:
    arch:       str = "vits8"   # 'vits8' | 'vitb8' | 'vits16' | 'vitb16'
    patch_size: int = 8


@dataclass
class TrainConfig:
    epochs:                 int   =  50
    learning_rate:          float = 5e-5
    weight_decay:           float = 0.05
    warmup_epochs:          int   = 4
    grad_clip:              float = 1.0
    next_index_loss_weight: float = 0.0
    device:                 str   = "cuda"
    seed:                   int   = 142
    log_interval:        int = 5   # steps between console logs
    checkpoint_interval: int = 1000   # steps between checkpoint saves
    val_interval:        int = 1000   # steps between validation runs
    val_num_samples:     int = 32    # images to visualise per validation run
    save_dir:            str = _os.path.join(_PROJECT_ROOT, "checkpoints")
    output_dir:          str = _os.path.join(_PROJECT_ROOT, "outputs")
    per_example_loss:    bool = False  # if True, average loss per example then over batch
    use_bf16:            bool = True   # bf16 autocast on A100/H100; no-op on CPU
    train_predict_next:  bool = False  # if True, fine-tune only PredictNext after main training

@dataclass
class EvalConfig:
    eval_batch_size = 32

@dataclass
class InferenceConfig:
    save_dir:    str   = _os.path.join(_PROJECT_ROOT, "outputs", "generated")
    n_samples:   int   = 32                    # number of images to generate per run
    temperature: float = 0.7                 # sampling temperature
    T:           int   = 8                   # number of unmasking steps

@dataclass
class Config:
    data:      DataConfig      = field(default_factory=DataConfig)
    model:     ModelConfig     = field(default_factory=ModelConfig)
    dino:      DinoConfig      = field(default_factory=DinoConfig)
    train:     TrainConfig     = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    

