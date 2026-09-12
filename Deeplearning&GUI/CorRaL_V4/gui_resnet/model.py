#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py: 하위 호환용 래퍼.
실제 모델 정의는 models.py를 사용한다.
"""
from models import GRID_SIZE, NUM_PIXELS, OUTPUT_DIM_DEFAULT, ResNet18_25x25, ResNet50_25x25, DenseNet25x25

__all__ = [
    "GRID_SIZE",
    "NUM_PIXELS",
    "OUTPUT_DIM_DEFAULT",
    "ResNet18_25x25",
    "ResNet50_25x25",
    "DenseNet25x25",
]

