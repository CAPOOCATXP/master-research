"""Frozen feature extraction.

The whole research design depends on the encoder and preprocessing being
**frozen**: if compression is allowed to change the feature space, the
decision-preservation question becomes unanswerable because there is no longer a
single reference detector to preserve.

This module provides the Week-1 backbone used for the environment check:

``wide_resnet50_2`` layer2+layer3, ImageNet-pretrained, ImageNet statistics
normalisation, bilinear resize to 256 then centre-crop to 224, patch size 3 with
stride 1, layer3 upsampled to layer2's spatial resolution and concatenated.

That configuration follows the *documented settings* of the official PatchCore
repository (see third_party/patchcore-inspection/README.md and
sample_training.sh). It is a faithful re-implementation of the feature
extractor only. The anomaly **score** implemented in this repository is the
plain max-nearest-distance detector described in Part A6, which is deliberately
NOT PatchCore's reweighted score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class BackboneSpec:
    """Everything that must be pinned for a feature space to be reproducible."""

    name: str
    weights: str
    layers: Tuple[str, ...]
    image_size: int
    resize: int
    patch_size: int
    patch_stride: int
    normalise_mean: Tuple[float, float, float]
    normalise_std: Tuple[float, float, float]

    def preprocessing_id(self) -> str:
        return (
            f"resize{self.resize}-center{self.image_size}-"
            f"ps{self.patch_size}-st{self.patch_stride}-"
            f"norm{'-'.join(f'{m:g}' for m in self.normalise_mean)}"
            f"_{'-'.join(f'{s:g}' for s in self.normalise_std)}"
        )

    def encoder_id(self) -> str:
        return f"{self.name}:{self.weights}:{'+'.join(self.layers)}"


# The PatchCore-documented configuration. Kept as a module constant so that a
# change is a visible edit rather than a scattered string.
WR50_LAYER23 = BackboneSpec(
    name="wide_resnet50_2",
    weights="IMAGENET1K_V1",
    layers=("layer2", "layer3"),
    image_size=224,
    resize=256,
    patch_size=3,
    patch_stride=1,
    normalise_mean=(0.485, 0.456, 0.406),
    normalise_std=(0.229, 0.224, 0.225),
)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def describe_device(requested: str = "auto"):
    """Resolve a torch device and report exactly what was chosen and why."""
    import torch

    notes = []
    if requested == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            notes.append("MPS reported available")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            notes.append("CUDA reported available")
        else:
            device = torch.device("cpu")
            notes.append("no accelerator available; CPU fallback")
    else:
        device = torch.device(requested)
    return device, notes


def image_to_tensor(image, resize: int, image_size: int, mean, std):
    """PIL image -> normalised NCHW float32 tensor (1, 3, S, S)."""
    import torch
    from torchvision.transforms import functional as TF

    if image.mode != "RGB":
        image = image.convert("RGB")
    t = TF.to_tensor(image)
    t = TF.resize(t, [resize, resize], antialias=True)
    t = TF.center_crop(t, [image_size, image_size])
    t = TF.normalize(t, mean=list(mean), std=list(std))
    return t.unsqueeze(0).to(torch.float32)


def load_backbone(spec: BackboneSpec = WR50_LAYER23):
    """Load the frozen pretrained backbone in eval mode with grads disabled."""
    import torch
    from torchvision.models import wide_resnet50_2
    from torchvision.models import Wide_ResNet50_2_Weights

    if spec.name != "wide_resnet50_2":
        raise NotImplementedError(f"backbone {spec.name} not wired up in Week 1")
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
    model = wide_resnet50_2(weights=weights)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, weights


def extract_patch_embeddings(model, batch, spec: BackboneSpec = WR50_LAYER23) -> np.ndarray:
    """Return ``(H*W, C)`` float32 patch embeddings for a single-image batch.

    layer3 is upsampled to layer2's spatial size with bilinear interpolation,
    then the two maps are concatenated channel-wise. Patch embeddings are read
    out with an unfold so that ``patch_size``/``patch_stride`` are explicit
    rather than implied by an average-pool kernel.
    """
    import torch
    import torch.nn.functional as F

    captured = {}

    def make_hook(name):
        def hook(_module, _inp, out):
            captured[name] = out

        return hook

    handles = []
    for name in spec.layers:
        handles.append(getattr(model, name).register_forward_hook(make_hook(name)))
    try:
        with torch.no_grad():
            model(batch)
    finally:
        for h in handles:
            h.remove()

    for name in spec.layers:
        if name not in captured:
            raise RuntimeError(f"layer {name} produced no output; check the backbone spec")

    l2 = captured[spec.layers[0]]
    l3 = captured[spec.layers[1]]
    if l3.shape[-2:] != l2.shape[-2:]:
        l3 = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
    feat = torch.cat([l2, l3], dim=1)  # (1, C, H, W)

    # Local neighbourhood aggregation (PatchCore uses avg_pool with the same
    # kernel/stride). Then unfold neighbourhoods and concatenate them.
    if spec.patch_size > 1:
        feat = F.avg_pool2d(feat, kernel_size=spec.patch_size, stride=spec.patch_stride, padding=0)
    unfolded = F.unfold(feat, kernel_size=1, stride=1)  # (1, C, L)
    patches = unfolded.squeeze(0).transpose(0, 1).contiguous()  # (L, C)
    return patches.detach().to("cpu").numpy().astype(np.float32)


def patch_grid_shape(height: int, width: int, spec: BackboneSpec = WR50_LAYER23, backbone_stride: int = 8):
    """Spatial shape of the patch grid, for reshaping scores back into a map.

    ``backbone_stride`` is 8 because layer2 of a ResNet has stride 8; layer3
    (stride 16) is upsampled to match it.
    """
    h = (height + backbone_stride - 1) // backbone_stride
    w = (width + backbone_stride - 1) // backbone_stride
    if spec.patch_size > 1:
        h = (h - spec.patch_size) // spec.patch_stride + 1
        w = (w - spec.patch_size) // spec.patch_stride + 1
    return h, w
