#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models.py: inverse_gui 공용 모델 정의 모음.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models
from torchvision.models import densenet121, DenseNet121_Weights


GRID_SIZE = 25
NUM_PIXELS = GRID_SIZE * GRID_SIZE
OUTPUT_DIM_DEFAULT = 273


class ResNet18_25x25(nn.Module):
    """25x25 binary layout용 ResNet18 forward surrogate."""

    def __init__(self, output_dim: int, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()

        if pretrained:
            try:
                self.resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
            except Exception:
                self.resnet = tv_models.resnet18(weights=None)
        else:
            self.resnet = tv_models.resnet18(weights=None)

        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

        old_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()

        if pretrained:
            with torch.no_grad():
                pretrained_weight = old_conv1.weight.mean(dim=1, keepdim=True)
                pretrained_weight = F.interpolate(
                    pretrained_weight,
                    size=(3, 3),
                    mode="bilinear",
                    align_corners=False,
                )
                self.resnet.conv1.weight.copy_(pretrained_weight)

        for param in self.resnet.conv1.parameters():
            param.requires_grad = True

        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
        )
        self._init_head()

    def _init_head(self) -> None:
        for module in self.resnet.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


class ResNet50_25x25(nn.Module):
    """25x25 binary layout용 ResNet50 forward surrogate.

    ResNet18 대비:
    - Bottleneck 블록 사용 (더 깊은 표현력)
    - Feature dimension 512 → 2048 (4배 확장)
    - Regression head를 2048→1024→512→output_dim으로 확장
    - 단계적 Dropout(0.3→0.2)으로 과적합 방지
    """

    def __init__(self, output_dim: int, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()

        if pretrained:
            try:
                self.resnet = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
            except Exception:
                self.resnet = tv_models.resnet50(weights=None)
        else:
            self.resnet = tv_models.resnet50(weights=None)

        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

        old_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()

        if pretrained:
            with torch.no_grad():
                pretrained_weight = old_conv1.weight.mean(dim=1, keepdim=True)
                pretrained_weight = F.interpolate(
                    pretrained_weight,
                    size=(3, 3),
                    mode="bilinear",
                    align_corners=False,
                )
                self.resnet.conv1.weight.copy_(pretrained_weight)

        for param in self.resnet.conv1.parameters():
            param.requires_grad = True

        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim),
        )
        self._init_head()

    def _init_head(self) -> None:
        for module in self.resnet.fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


class DenseNet25x25(nn.Module):
    """25x25 binary layout용 DenseNet121 forward surrogate."""

    def __init__(
        self,
        output_dim: int = OUTPUT_DIM_DEFAULT,
        use_pretrained: bool = True,
        pretrained_path=None,
        stem_type: str = "standard",
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.stem_type = stem_type

        # 기본 백본 생성 (훈련 코드 구조와 동기화)
        base = densenet121(weights=None)

        if stem_type == "small":
            base.features.conv0 = nn.Conv2d(1, 32, 3, 1, 1, bias=False)
            base.features.norm0 = nn.BatchNorm2d(32)
            base.features.pool0 = nn.Sequential(
                nn.Conv2d(32, 64, 3, 1, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        elif stem_type == "cbam":
            # [최신] train_add_m.py의 CBAM 삽입 구조
            base.features.conv0 = nn.Conv2d(1, 32, 3, stride=1, padding=1, bias=False)
            base.features.norm0 = nn.BatchNorm2d(32)
            base.features.pool0 = nn.Sequential(
                nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            
            new_features = OrderedDict()
            from collections import OrderedDict as _ODict
            for name, module in base.features.named_children():
                new_features[name] = module
                if name == 'denseblock1':
                    new_features['cbam1'] = CBAM(256)
                elif name == 'denseblock2':
                    new_features['cbam2'] = CBAM(512)
                elif name == 'denseblock3':
                    new_features['cbam3'] = CBAM(1024)
                elif name == 'denseblock4':
                    new_features['cbam4'] = CBAM(1024)
            base.features = nn.Sequential(new_features)
        else:
            # Standard: 7x7 stride-1 + Identity pool
            base.features.conv0 = nn.Conv2d(1, 64, 7, 1, 3, bias=False)
            base.features.pool0 = nn.Identity()

        self.densenet = base

        if freeze_backbone:
            for p in self.densenet.parameters():
                p.requires_grad = False

        if pretrained_path:
            # 보조적인 가중치 로드 로직
            try:
                ck = torch.load(pretrained_path, map_location="cpu", weights_only=False)
                sd = ck.get("model", ck.get("state_dict", ck))
                self.densenet.load_state_dict(sd, strict=False)
            except Exception: pass

        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, output_dim),
            nn.Sigmoid(),
        )
        self._init_fc()

    def _init_fc(self):
        for module in self.densenet.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.densenet.features(x)
        out = torch.nn.functional.relu(feats, inplace=True)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        return self.densenet.classifier(torch.flatten(out, 1))

# ── CBAM (Convolutional Block Attention Module) ──────────────────────────────
class _ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        avg_pool = x.mean(dim=(2, 3))
        max_pool = x.amax(dim=(2, 3))
        attn = torch.sigmoid(self.shared_mlp(avg_pool) + self.shared_mlp(max_pool))
        return x * attn.view(b, c, 1, 1)

class _SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = _ChannelAttention(channels, reduction)
        self.sa = _SpatialAttention(spatial_kernel)
    def forward(self, x):
        return self.sa(self.ca(x))
