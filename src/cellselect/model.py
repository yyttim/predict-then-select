"""Reference CNN--ViT encoder with independent marker-mean and binary heads.

Adapted from Saqib Nazir and Ardhendu Behera's MIT-licensed
Single-Cell-Phenotyping, commit 56a42e77bf6a7c5fcbbf8967dc209e2ee8f3d5df.
See vendor/LICENSE and vendor/README.md. The reference encoder is preserved;
cross-task gating is removed. The two heads predict standardized marker mean
and the logit of a fixed high-marker event.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class EfficientChannelAttention(nn.Module):
    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        weights = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.conv(weights).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(weights)


class InceptionModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        first = second = out_channels // 3
        third = out_channels - first - second
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, first, 1, bias=False),
            nn.BatchNorm2d(first),
            nn.GELU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, second, 3, padding=1, bias=False),
            nn.BatchNorm2d(second),
            nn.GELU(),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, third, 3, padding=1, bias=False),
            nn.BatchNorm2d(third),
            nn.GELU(),
            nn.Conv2d(third, third, 3, padding=1, bias=False),
            nn.BatchNorm2d(third),
            nn.GELU(),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        branches = (self.branch1(x), self.branch2(x), self.branch3(x))
        return torch.cat(branches, dim=1) + self.shortcut(x)


class CNNBackbone(nn.Module):
    feature_dim = 192

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout2d(0.1),
        )
        self.inception1 = InceptionModule(64, 128)
        self.transition = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout2d(0.1),
        )
        self.inception2 = InceptionModule(128, 192)

    def forward(self, x: Tensor) -> Tensor:
        x = self.inception1(self.stem(x))
        x = self.inception2(self.transition(x))
        return x.flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * dim, dim),
            nn.Dropout(0.1),
        )

    def forward(self, x: Tensor) -> Tensor:
        normalized = self.norm1(x)
        # No downstream computation uses attention maps. This enables PyTorch's
        # optimized attention path without changing the architecture.
        attended, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        x = x + attended
        return x + self.mlp(self.norm2(x))


class ViTBackbone(nn.Module):
    embed_dim = 128

    def __init__(self, img_size: int = 28, in_channels: int = 4) -> None:
        super().__init__()
        self.patch_size = 4
        self.num_patches = (img_size // self.patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([TransformerBlock(self.embed_dim, 4) for _ in range(2)])
        self.norm = nn.LayerNorm(self.embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(batch_size, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class FeatureFusion(nn.Module):
    """Scalar-weighted addition of global CNN and ViT CLS projections."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.cnn_proj = nn.Linear(CNNBackbone.feature_dim, feature_dim)
        self.vit_proj = nn.Linear(ViTBackbone.embed_dim, feature_dim)
        self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, cnn: Tensor, vit: Tensor) -> Tensor:
        weights = F.softmax(self.fusion_weight, dim=0)
        cnn = self.cnn_proj(cnn.mean(dim=1))
        vit = self.vit_proj(vit[:, 0])
        return self.norm(weights[0] * cnn + weights[1] * vit)


class SharedEncoder(nn.Module):
    """Reference hybrid encoder with all parameters trainable."""

    def __init__(self, feature_dim: int, img_size: int, in_channels: int) -> None:
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.shared_channel_attention = EfficientChannelAttention()
        self.cnn = CNNBackbone(in_channels)
        self.vit = ViTBackbone(img_size, in_channels)
        self.fusion = FeatureFusion(feature_dim)

    def forward(self, x: Tensor) -> Tensor:
        expected = (self.in_channels, self.img_size, self.img_size)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected input [B, {expected}], received {tuple(x.shape)}")
        if x.shape[0] == 0:
            raise ValueError("An input batch must contain at least one cell")
        if not x.is_floating_point():
            raise TypeError("DPC inputs must be floating point after preprocessing")
        x = self.shared_channel_attention(x)
        return self.fusion(self.cnn(x), self.vit(x))


class TaskSpecificRefinement(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.refinement = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.refinement(x)


class IndependentHead(nn.Module):
    def __init__(self, feature_dim: int, output_dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.refinement = TaskSpecificRefinement(feature_dim)
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.predictor(self.refinement(features))


class MarkerModel(nn.Module):
    """Both outputs have shape [B]; inputs are normalized [B, 4, 28, 28]."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_dim = 256
        self.encoder = SharedEncoder(256, 28, 4)
        self.mean_head = IndependentHead(256, 1, 0.4)
        self.hit_head = IndependentHead(256, 1, 0.4)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        features = self.encoder(x)
        # All objectives execute both heads in this order, preserving paired RNG streams.
        return {
            "mean_z": self.mean_head(features).squeeze(-1),
            "hit_logit": self.hit_head(features).squeeze(-1),
        }


def build_model(seed: int | None = None) -> MarkerModel:
    """Seed CPU initialization without changing the caller's RNG state."""
    if seed is None:
        return MarkerModel()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return MarkerModel()


def parameter_counts(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
