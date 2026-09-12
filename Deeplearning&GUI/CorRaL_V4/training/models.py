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

        # ── 입력 적응: 1ch, 25x25 소형 입력 대응 ──
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

        # ── Regression Head: 2048 → 1024 → 512 → output_dim ──
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

        load_imagenet = use_pretrained and (stem_type == "standard") and (pretrained_path is None)
        if load_imagenet:
            try:
                self.densenet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            except Exception:
                self.densenet = densenet121(weights=None)
        else:
            self.densenet = densenet121(weights=None)

        if freeze_backbone:
            for p in self.densenet.parameters():
                p.requires_grad = False

        if stem_type == "small":
            self.densenet.features.conv0 = nn.Conv2d(1, 32, 3, 1, 1, bias=False)
            self.densenet.features.norm0 = nn.BatchNorm2d(32)
            self.densenet.features.pool0 = nn.Sequential(
                nn.Conv2d(32, 64, 3, 1, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        else:
            old_conv = self.densenet.features.conv0
            new_conv = nn.Conv2d(1, 64, 7, 1, 3, bias=False)
            if load_imagenet:
                with torch.no_grad():
                    new_conv.weight = nn.Parameter(old_conv.weight.mean(dim=1, keepdim=True))
            self.densenet.features.conv0 = new_conv
            self.densenet.features.pool0 = nn.Identity()
            for p in self.densenet.features.conv0.parameters():
                p.requires_grad = True

        if pretrained_path:
            try:
                try:
                    ck = torch.load(pretrained_path, map_location="cpu", weights_only=True)
                except Exception:
                    ck = torch.load(pretrained_path, map_location="cpu", weights_only=False)
                sd = ck.get("model", ck.get("state_dict", ck))
                self.densenet.load_state_dict(sd, strict=False)
            except Exception:
                pass

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
