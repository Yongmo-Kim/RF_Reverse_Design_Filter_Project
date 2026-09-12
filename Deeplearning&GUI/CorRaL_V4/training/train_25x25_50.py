#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_25x25_50.py: 25x25 RF 필터 Forward Model 학습 코드 (ResNet50) — H100 최적화판

ResNet18(train_25x25.py) 대비 주요 차이:
  - Backbone: ResNet18 → ResNet50 (Bottleneck, 2048-dim feature)
  - Regression Head: 2048→1024→512→output_dim (더 넓은 head)
  - 깊고 넓은 네트워크 → 더 긴 수렴 시간 → early-stop / walltime 예산 중시

train_add_m.py (DenseNet H100 판)에서 가져온 H100 최적화 항목:
  1. auto_batch_size(): VRAM 용량으로 batch_size 자동 결정
     (수동 --batch-size 지정 시 그 값을 우선 사용)
  2. TF32 / cudnn.benchmark / matmul precision=high (Hopper / Ampere 성능 활성화)
  3. bfloat16 autocast (compute capability ≥ 8.0) / fp16 fallback
  4. channels_last memory format (ResNet 계열에서 유의미한 속도 향상)
  5. persistent_workers / prefetch_factor / pin_memory 활용 DataLoader
  6. _save_state_dict_atomic(): NFS 환경 안전 저장
  7. accum_steps 동적 결정 (batch_size ≥ 1024 이면 1, 그 외 2)
  8. Windows 환경에서 num_workers=0 강제

기존 유지 기능:
  - 5-Fold CV (Stage 1) + 95/5 holdout 최종 학습 (Stage 2)
  - --max-walltime-hours 기반 시간 예산 관리 (기본 9.5h)
  - BatchNorm running stats 재보정
  - ImageNet pretrained 기본 사용 (--no-pretrained 로 끌 수 있음)
  - 8채널(S11/S12/S21/S22 × dB+deg) 출력

train_25x25.py(ResNet18 똑똑한 학습판) 에서 가져온 학습 품질 향상 항목:
  - EMA (Exponential Moving Average): 학습 weight 의 평활본 평가
  - ChannelWeightedSmoothL1Loss: 채널별 가중 (S21/S12_db 1.5, deg 0.7)
  - CosineAnnealingWarmRestarts: T_0=patience*steps, T_mult=2 로 주기적 LR 재상승
  - EMA-vs-raw best state 선택: 두 모델을 매 epoch 평가, 더 좋은 쪽을 best 로 채택
  - (제외) 학습 시 flip augmentation: 데이터셋이 이미 4× 오프라인 증강되어 있어 미적용
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from models import GRID_SIZE, ResNet50_25x25


TITLE = "=" * 72
SEP = "-" * 72
SEED = 42
K_FOLDS = 5

# 변경: batch_size 가 클 때는 accumulation 을 쓰지 않아도 되도록 동적으로 결정 (main 에서 조정)
DEFAULT_ACCUMULATION_STEPS = 2

# 변경: H100 자동 배치 사이즈 탐지를 우선시하므로 기본값을 None 으로 두고
#        사용자가 --batch-size 를 지정하면 수동 모드로 전환
DEFAULT_BATCH_SIZE = None
DEFAULT_MAX_EPOCHS = 200
DEFAULT_MAX_LR = 5e-5          # ResNet50 + AdamW + CosineAnnealingWarmRestarts 안정 수렴용
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_PATIENCE = 35

CHECKPOINT_PATTERN = "forward_fold{fold}.pt"
FINAL_CHECKPOINT_NAME = "forward_final.pt"
DB_KEYS = [
    "s11_db", "s12_db", "s21_db", "s22_db",
    "s11_deg", "s12_deg", "s21_deg", "s22_deg",
]
# 변경: S21/S12 통과 dB 채널은 필터 성능의 주축이므로 가중을 살짝 더 부여
#        (train_25x25.py 와 동일한 가중치)
CHANNEL_LOSS_WEIGHTS = {
    "s11_db": 1.0, "s12_db": 1.5, "s21_db": 1.5, "s22_db": 1.0,
    "s11_deg": 0.7, "s12_deg": 0.7, "s21_deg": 0.7, "s22_deg": 0.7,
}

# 변경: H100 환경에서는 DataLoader 워커를 늘려 GPU 유휴 시간 최소화
DEFAULT_NUM_WORKERS = 8
DEFAULT_MAX_WALLTIME_HOURS = 9.5

# 변경: EMA decay (높을수록 부드러움). ResNet50 + 6만~10만 샘플 기준 0.9995 적정
#        (ResNet18 보다 파라미터가 약 2배 많아 더 강한 평활 권장)
DEFAULT_EMA_DECAY = 0.9995


# ──────────────────────────────────────────────────────────────────────────────
# H100 / 일반 GPU 환경 자동 튜닝
# ──────────────────────────────────────────────────────────────────────────────
def configure_backend_for_h100() -> None:
    """H100 / A100 급 GPU 에서 TF32 / cudnn.benchmark / matmul precision 을 활성화.

    - torch.set_float32_matmul_precision('high') : TF32 기반 matmul 허용
    - cuda.matmul.allow_tf32 / cudnn.allow_tf32  : 내부 연산도 TF32 허용
    - cudnn.benchmark = True                     : 고정 입력 크기에서 최적 알고리즘 선택
    """
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def pick_autocast_dtype(device: torch.device) -> torch.dtype:
    """Ampere / Hopper 급 GPU 에서는 bfloat16, 그 외는 fp16 fallback."""
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability(device)
            if major >= 8:
                return torch.bfloat16
        except Exception:
            pass
    return torch.float16


def auto_batch_size(device: torch.device, manual_batch: int | None = None) -> int:
    """
    VRAM 기반 batch_size 자동 결정 (ResNet50 + 25x25 입력 기준).

      ≥ 80 GB (H100 / A100 80G)  → 1024
      ≥ 40 GB (A100 40G)         →  512
      ≥ 20 GB (A6000 / 3090 24G) →  256
      ≥ 10 GB                    →  128
      ≥  6 GB                    →   64
      CPU / < 6 GB               →   32
    """
    if manual_batch is not None:
        print(f"  [Batch] 수동 설정: {manual_batch}")
        return int(manual_batch)

    if not torch.cuda.is_available():
        bs = 32
        print(f"  [Batch] CPU 모드 → {bs}")
        return bs

    vram_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    if   vram_gb >= 80: bs = 1024
    elif vram_gb >= 40: bs = 512
    elif vram_gb >= 20: bs = 256
    elif vram_gb >= 10: bs = 128
    elif vram_gb >=  6: bs = 64
    else:               bs = 32

    print(f"  [Batch] VRAM {vram_gb:.0f} GB → 자동 batch_size={bs}")
    return bs


def resolve_accum_steps(batch_size: int) -> int:
    """
    큰 batch 에서는 gradient accumulation 없이 한 번에 step.
    batch_size >= 1024 이면 accum=1, 아니면 2 (effective batch 2배).
    """
    return 1 if batch_size >= 1024 else DEFAULT_ACCUMULATION_STEPS


def resolve_num_workers(requested: int) -> int:
    """Windows 는 num_workers=0 이 안전. 그 외에는 cpu_count 를 상한으로 사용."""
    if os.name == "nt":
        return 0
    try:
        cap = os.cpu_count() or 4
    except Exception:
        cap = 4
    return max(0, min(int(requested), cap))


# ──────────────────────────────────────────────────────────────────────────────
# 체크포인트 / 유틸
# ──────────────────────────────────────────────────────────────────────────────
def _save_state_dict_atomic(state_dict, path: Path) -> None:
    """NFS 등에서 저장 도중 깨짐 방지: 임시 파일 후 rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state_dict, tmp)
    os.replace(str(tmp), str(path))


def set_all_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    def __init__(self, patience: int = DEFAULT_PATIENCE, min_delta: float = 1e-5, verbose: bool = False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float("inf")
        self.best_model = None
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"    ★ New best model @ epoch {epoch}: {val_loss:.6f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


# ──────────────────────────────────────────────────────────────────────────────
# EMA (Exponential Moving Average) of model weights
# ──────────────────────────────────────────────────────────────────────────────
class EMA:
    """학습 중 매 step 마다 ema_param = decay * ema_param + (1-decay) * param.

    학습 종료 시 EMA weights 로 평가 → SWA 와 비슷한 generalization 효과.
    BatchNorm running stats 등 정수형은 그대로 복사.
    """

    def __init__(self, model: nn.Module, decay: float = DEFAULT_EMA_DECAY):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
                else:
                    self.shadow[k] = v.detach().clone()

    def state_dict(self) -> dict:
        return {k: v.detach().clone() for k, v in self.shadow.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Channel-weighted SmoothL1Loss
# ──────────────────────────────────────────────────────────────────────────────
class ChannelWeightedSmoothL1Loss(nn.Module):
    """채널별 가중 SmoothL1.

    target shape: (B, len(DB_KEYS) * F)
    각 채널 블록에 CHANNEL_LOSS_WEIGHTS 의 가중을 적용한 후 평균.
    채널 분리가 불가하면 일반 SmoothL1 으로 fallback.
    """

    def __init__(self, channel_weights: dict[str, float], db_keys: list[str], beta: float = 1.0):
        super().__init__()
        self.channel_weights = channel_weights
        self.db_keys = db_keys
        self.beta = beta
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none", beta=beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_keys = len(self.db_keys)
        if pred.shape[1] % n_keys != 0:
            return self.smooth_l1(pred, target).mean()
        f = pred.shape[1] // n_keys
        per_elem = self.smooth_l1(pred, target)  # (B, n_keys * F)
        weights = torch.tensor(
            [self.channel_weights.get(k, 1.0) for k in self.db_keys],
            device=pred.device, dtype=per_elem.dtype,
        )  # (n_keys,)
        per_elem = per_elem.view(pred.shape[0], n_keys, f)
        per_channel_mean = per_elem.mean(dim=(0, 2))  # (n_keys,)
        weighted = (per_channel_mean * weights).sum() / weights.sum().clamp_min(1e-8)
        return weighted


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 경로 / 로딩
# ──────────────────────────────────────────────────────────────────────────────
def default_data_paths() -> tuple[Path, Path]:
    if os.name == "nt":
        base = Path(r"C:\Users\User\OneDrive - 금오공과대학교\박강현의 파일 - 02. 연구 자료 공유용\12. Preprocess\dataset_Renewal_augmented_2")
    else:
        base = Path("/scratch/home/bhj101500/25x25_data")
    return base / "layouts.npz", base / "s_params.npz"


def default_output_dir() -> Path:
    if os.name == "nt":
        return Path(r"C:\Users\User\OneDrive - 금오공과대학교\바탕 화면\ResNet50코드\outputs")
    return Path("/scratch/home/bhj101500/25x25_outputs_r50")


def load_training_data(layouts_path: Path, sparams_path: Path) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, dict]:
    layouts_data = np.load(layouts_path, allow_pickle=True)
    sparams_data = np.load(sparams_path, allow_pickle=True)

    if "layouts" not in layouts_data.files:
        raise KeyError("layouts.npz에 'layouts' 키가 없습니다.")
    missing = [key for key in DB_KEYS + ["frequencies"] if key not in sparams_data.files]
    if missing:
        raise KeyError(f"s_params.npz에 필요한 키가 없습니다: {missing}")

    layouts = layouts_data["layouts"].astype(np.float32)
    frequencies = sparams_data["frequencies"].astype(np.float32)

    if layouts.shape[1:] != (GRID_SIZE, GRID_SIZE):
        raise ValueError(f"입력 레이아웃 크기가 {GRID_SIZE}x{GRID_SIZE}가 아닙니다: {layouts.shape}")

    arrays = []
    normalization = {}
    sample_counts = [len(layouts)]
    for key in DB_KEYS:
        values = sparams_data[key].astype(np.float32)
        sample_counts.append(len(values))
        value_min = float(values.min())
        value_max = float(values.max())
        normalization[key] = {"min": value_min, "max": value_max}
        arrays.append((values - value_min) / (value_max - value_min + 1e-8))

    if len(set(sample_counts)) != 1:
        raise ValueError(f"샘플 수가 서로 일치하지 않습니다: {sample_counts}")

    targets = np.concatenate(arrays, axis=1).astype(np.float32)
    layouts = (layouts > 0.5).astype(np.float32)

    X = torch.from_numpy(layouts).unsqueeze(1)
    y = torch.from_numpy(targets)

    metadata = {
        "target_keys": DB_KEYS,
        "normalization": normalization,
        "freq_points": int(len(frequencies)),
        "freq_range_ghz": [float(frequencies[0]), float(frequencies[-1])],
    }
    return X, y, frequencies, metadata


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(param.numel() for param in model.parameters() if param.requires_grad)
    return sum(param.numel() for param in model.parameters())


def make_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory, drop_last):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True   # worker 재생성 오버헤드 제거
        kwargs["prefetch_factor"] = 4         # H100 기준으로 prefetch 를 2 → 4 로 상향
    return DataLoader(dataset, **kwargs)


def elapsed_hours(start_time: float) -> float:
    return (time.time() - start_time) / 3600.0


def time_budget_exceeded(start_time: float, max_walltime_hours: float) -> bool:
    return elapsed_hours(start_time) >= max_walltime_hours


# ──────────────────────────────────────────────────────────────────────────────
# 학습 / 평가 루프
# ──────────────────────────────────────────────────────────────────────────────
def _to_device_channels_last(tensor: torch.Tensor, device: torch.device, use_channels_last: bool) -> torch.Tensor:
    """입력을 device 로 옮긴 뒤, channels_last 옵션이 켜져 있으면 memory_format 변경."""
    tensor = tensor.to(device, non_blocking=True)
    if use_channels_last and tensor.dim() == 4:
        tensor = tensor.to(memory_format=torch.channels_last)
    return tensor


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    scaler,
    device,
    accumulation_steps: int = DEFAULT_ACCUMULATION_STEPS,
    amp_dtype: torch.dtype = torch.float16,
    use_channels_last: bool = False,
    ema: "EMA | None" = None,
    scheduler=None,
):
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    num_batches = len(dataloader)
    progress = tqdm(dataloader, desc="Training", leave=False)
    use_cuda = device.type == "cuda"
    for step, (images, targets) in enumerate(progress):
        images = _to_device_channels_last(images, device, use_channels_last)
        targets = targets.to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda):
            outputs = model(images)
            loss = criterion(outputs, targets) / accumulation_steps
        scaler.scale(loss).backward()
        is_accumulation_step = (step + 1) % accumulation_steps == 0
        is_last_batch = (step + 1) == num_batches
        if is_accumulation_step or is_last_batch:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # 변경: optimizer.step 직후 EMA shadow 업데이트
            if ema is not None:
                ema.update(model)
            # 변경: CosineAnnealingWarmRestarts 는 step 단위로 호출 (epoch 단위로는 사용 안 함)
            if scheduler is not None:
                try:
                    scheduler.step()
                except (ValueError, ZeroDivisionError):
                    pass
        total_loss += loss.item() * accumulation_steps
        progress.set_postfix(loss=f"{loss.item() * accumulation_steps:.4f}")
    return total_loss / max(num_batches, 1)


def evaluate(model, dataloader, criterion, device, amp_dtype: torch.dtype = torch.float16, use_channels_last: bool = False):
    model.eval()
    total_loss = 0.0
    use_cuda = device.type == "cuda"
    with torch.no_grad():
        for images, targets in dataloader:
            images = _to_device_channels_last(images, device, use_channels_last)
            targets = targets.to(device, non_blocking=True)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_cuda):
                outputs = model(images)
                loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)


def evaluate_with_state_dict(model, state_dict, dataloader, criterion, device,
                             amp_dtype: torch.dtype, use_channels_last: bool) -> float:
    """주어진 state_dict 를 임시로 적용해 평가하고 원본은 보존 (EMA 평가용)."""
    backup = copy.deepcopy(model.state_dict())
    try:
        model.load_state_dict(state_dict)
        return evaluate(model, dataloader, criterion, device, amp_dtype, use_channels_last)
    finally:
        model.load_state_dict(backup)


def recalibrate_bn(model, dataloader, device, use_channels_last: bool = False):
    """best_state 복원 후 BatchNorm running statistics 를 현재 도메인 데이터로 재계산."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()
            m.train()
    with torch.no_grad():
        for images, _ in dataloader:
            images = _to_device_channels_last(images, device, use_channels_last)
            model(images)
    model.eval()


def build_model(output_dim: int, pretrained: bool, device: torch.device, use_channels_last: bool) -> ResNet50_25x25:
    """ResNet50_25x25 모델을 생성하고 device + (옵션) channels_last 에 올린다."""
    model = ResNet50_25x25(output_dim=output_dim, pretrained=pretrained, freeze_backbone=False)
    model = model.to(device)
    if use_channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: K-Fold Cross Validation (CosineAnnealingWarmRestarts + EarlyStopping + EMA)
# ──────────────────────────────────────────────────────────────────────────────
def _make_holdout_indices(num_samples: int, holdout_ratio: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Stage 2 에서 사용할 고정 validation holdout index 를 생성한다."""
    indices = np.arange(num_samples)
    rng = np.random.default_rng(SEED)
    rng.shuffle(indices)
    val_count = max(1, int(round(num_samples * holdout_ratio)))
    val_idx = np.sort(indices[:val_count])
    train_idx = np.sort(indices[val_count:])
    return train_idx, val_idx


def kfold_find_optimal_epochs(
    X, y, device, output_dir,
    max_epochs, batch_size, max_lr, weight_decay, patience,
    pretrained, num_workers, start_time, max_walltime_hours,
    amp_dtype: torch.dtype, use_channels_last: bool,
    ema_decay: float,
):
    print(TITLE)
    print("STAGE 1: 5-Fold Cross Validation (ResNet50, 똑똑한 학습판 + H100 최적화)")
    print(TITLE)

    ensemble_dir = output_dir / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    indices = np.arange(len(X))
    np.random.seed(SEED)
    np.random.shuffle(indices)
    fold_size = len(indices) // K_FOLDS
    output_dim = y.shape[1]
    accum_steps = resolve_accum_steps(batch_size)
    fold_results = []

    for fold in range(K_FOLDS):
        if time_budget_exceeded(start_time, max_walltime_hours):
            print(f"\nTime budget reached before fold {fold + 1}; stopping Stage 1 early.")
            break

        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < K_FOLDS - 1 else len(indices)
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        print(f"\n[Fold {fold + 1}/{K_FOLDS}] Train={len(train_idx):,}, Val={len(val_idx):,}"
              f"  batch={batch_size}  nw={num_workers}  accum={accum_steps}")

        train_loader = make_dataloader(
            TensorDataset(X[train_idx], y[train_idx]),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=device.type == "cuda", drop_last=True,
        )
        val_loader = make_dataloader(
            TensorDataset(X[val_idx], y[val_idx]),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=device.type == "cuda", drop_last=False,
        )

        model = build_model(output_dim=output_dim, pretrained=pretrained,
                            device=device, use_channels_last=use_channels_last)
        # 변경: SmoothL1Loss → ChannelWeightedSmoothL1Loss (S21/S12_db 가중 1.5)
        criterion = ChannelWeightedSmoothL1Loss(CHANNEL_LOSS_WEIGHTS, DB_KEYS)
        optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
        # 변경: CosineAnnealingLR(epoch step) → CosineAnnealingWarmRestarts(per optimizer step)
        steps_per_epoch = max(1, math.ceil(len(train_loader) / accum_steps))
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, patience * steps_per_epoch), T_mult=2, eta_min=1e-6,
        )
        scaler = GradScaler(device.type, enabled=device.type == "cuda")
        stopper = EarlyStopping(patience=patience, min_delta=1e-5, verbose=False)
        # 변경: EMA shadow 초기화
        ema = EMA(model, decay=ema_decay)

        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, device,
                accumulation_steps=accum_steps,
                amp_dtype=amp_dtype,
                use_channels_last=use_channels_last,
                ema=ema,
                scheduler=scheduler,
            )
            # 변경: raw 와 EMA 둘 다 평가하고 더 좋은 쪽을 val_loss 로 채택 → early stop 정밀도↑
            val_loss_raw = evaluate(
                model, val_loader, criterion, device,
                amp_dtype=amp_dtype, use_channels_last=use_channels_last,
            )
            val_loss_ema = evaluate_with_state_dict(
                model, ema.state_dict(), val_loader, criterion, device,
                amp_dtype, use_channels_last,
            )
            val_loss = min(val_loss_raw, val_loss_ema)
            if epoch == 1 or epoch % 20 == 0:
                print(f"    Epoch {epoch:3d} | Train {train_loss:.6f} | Val(raw) {val_loss_raw:.6f} | Val(ema) {val_loss_ema:.6f}")
            if stopper(val_loss, model, epoch):
                print(f"    Early stopped at epoch {epoch}")
                break
            if time_budget_exceeded(start_time, max_walltime_hours):
                print(f"    Time budget reached at fold {fold + 1}, epoch {epoch}; saving best-so-far and exiting Stage 1.")
                break

        if stopper.best_model is not None:
            model.load_state_dict(stopper.best_model)

        ckpt_name = CHECKPOINT_PATTERN.format(fold=fold)
        _save_state_dict_atomic(model.state_dict(), ensemble_dir / ckpt_name)
        fold_results.append({
            "fold": fold,
            "checkpoint": f"ensemble/{ckpt_name}",
            "best_epoch": stopper.best_epoch,
            "best_val_loss": float(stopper.best_loss),
        })
        print(f"    Saved {ckpt_name} | best epoch={stopper.best_epoch}, val={stopper.best_loss:.6f}")

        del model, optimizer, scheduler, scaler, ema
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if time_budget_exceeded(start_time, max_walltime_hours):
            break

    if fold_results:
        best_epochs = [result["best_epoch"] for result in fold_results]
        # 변경: ResNet18 과 동일한 경향식 (median + patience//2) 으로 안정화
        optimal_epochs = min(int(np.median(best_epochs)) + patience // 2, max_epochs)
    else:
        best_epochs = [max(1, min(patience, max_epochs))]
        optimal_epochs = best_epochs[0]

    print(SEP)
    print(f"Best epochs per fold: {best_epochs}")
    print(f"K-Fold scheduler = CosineAnnealingWarmRestarts(T_0={patience}*steps, T_mult=2)")
    print(f"Final training epochs = median(best_epochs) + patience//2 = {optimal_epochs}")
    return optimal_epochs, fold_results


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Final Training (95/5 holdout + EarlyStopping + EMA + EMA-vs-raw best 선택)
# ──────────────────────────────────────────────────────────────────────────────
def final_train(
    X, y, device, output_dir,
    optimal_epochs, batch_size, max_lr, weight_decay, patience,
    pretrained, num_workers, start_time, max_walltime_hours,
    amp_dtype: torch.dtype, use_channels_last: bool,
    ema_decay: float,
):
    print(TITLE)
    print("STAGE 2: Final Training - 95% Train / 5% Val Holdout (ResNet50, 똑똑한 학습판)")
    print(TITLE)

    accum_steps = resolve_accum_steps(batch_size)
    train_idx, val_idx = _make_holdout_indices(len(X), holdout_ratio=0.05)

    train_loader = make_dataloader(
        TensorDataset(X[train_idx], y[train_idx]),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=device.type == "cuda", drop_last=False,
    )
    val_loader = make_dataloader(
        TensorDataset(X[val_idx], y[val_idx]),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda", drop_last=False,
    )
    bn_loader = make_dataloader(
        TensorDataset(X[train_idx], y[train_idx]),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda", drop_last=False,
    )

    output_dim = y.shape[1]
    model = build_model(output_dim=output_dim, pretrained=pretrained,
                        device=device, use_channels_last=use_channels_last)
    # 변경: SmoothL1Loss → ChannelWeightedSmoothL1Loss
    criterion = ChannelWeightedSmoothL1Loss(CHANNEL_LOSS_WEIGHTS, DB_KEYS)
    optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    # 변경: CosineAnnealingLR → CosineAnnealingWarmRestarts (per optimizer step)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accum_steps))
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, patience * steps_per_epoch), T_mult=2, eta_min=1e-6,
    )
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    stopper = EarlyStopping(patience=patience, min_delta=1e-5, verbose=False)
    # 변경: EMA shadow 초기화
    ema = EMA(model, decay=ema_decay)

    best_train_loss = float("inf")
    best_eval_loss = float("inf")
    best_epoch = 0
    best_state = None
    # 변경: history 에 val_ema 컬럼 추가
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_ema": [], "lr": []}

    print(f"Training samples   : {len(train_idx):,}")
    print(f"Validation samples : {len(val_idx):,}")
    print(f"Planned max epochs : {optimal_epochs}")
    print(f"Accumulation steps : {accum_steps}")
    print(f"Total params       : {count_params(model, trainable_only=False):,}")
    print(f"Trainable params   : {count_params(model, trainable_only=True):,}")
    print(f"EMA decay          : {ema_decay}")
    print(f"Remaining walltime : {max(max_walltime_hours - elapsed_hours(start_time), 0.0):.2f} hours")

    for epoch in range(1, optimal_epochs + 1):
        if time_budget_exceeded(start_time, max_walltime_hours):
            print(f"Time budget reached before epoch {epoch}; stopping Stage 2 early.")
            break

        current_lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            accumulation_steps=accum_steps,
            amp_dtype=amp_dtype,
            use_channels_last=use_channels_last,
            ema=ema,
            scheduler=scheduler,  # CosineAnnealingWarmRestarts: optimizer step 단위
        )
        # 변경: raw + EMA 둘 다 평가 → 더 좋은 쪽을 best_state 로 채택
        val_loss_raw = evaluate(
            model, val_loader, criterion, device,
            amp_dtype=amp_dtype, use_channels_last=use_channels_last,
        )
        val_loss_ema = evaluate_with_state_dict(
            model, ema.state_dict(), val_loader, criterion, device,
            amp_dtype, use_channels_last,
        )
        val_loss = min(val_loss_raw, val_loss_ema)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss_raw)
        history["val_ema"].append(val_loss_ema)
        history["lr"].append(current_lr)

        if train_loss < best_train_loss:
            best_train_loss = train_loss
        if val_loss < best_eval_loss:
            best_eval_loss = val_loss
            best_epoch = epoch
            # 변경: EMA 가 더 좋으면 EMA state, 아니면 raw state 를 best 로 채택
            best_state = ema.state_dict() if val_loss_ema <= val_loss_raw else copy.deepcopy(model.state_dict())

        if best_train_loss < float("inf") and train_loss > best_train_loss * 10.0:
            print(f"⚠ Warning: epoch {epoch} train_loss={train_loss:.6f} is >10x best_loss={best_train_loss:.6f}")
        if epoch <= 10 or epoch % 10 == 0 or epoch == optimal_epochs:
            print(f"Epoch {epoch:3d} | Train {train_loss:.6f} | Val(raw) {val_loss_raw:.6f} | "
                  f"Val(ema) {val_loss_ema:.6f} | BestVal {best_eval_loss:.6f} | LR {current_lr:.2e}")

        if stopper(val_loss, model, epoch):
            print(f"Early stopped at epoch {epoch}")
            break
        if time_budget_exceeded(start_time, max_walltime_hours):
            print(f"Time budget reached at epoch {epoch}; loading best checkpoint and finishing Stage 2.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("Recalibrating BatchNorm running statistics...")
    recalibrate_bn(model, bn_loader, device, use_channels_last=use_channels_last)

    final_loss = evaluate(
        model, val_loader, criterion, device,
        amp_dtype=amp_dtype, use_channels_last=use_channels_last,
    )
    _save_state_dict_atomic(model.state_dict(), output_dir / FINAL_CHECKPOINT_NAME)
    final_info = {
        "final_checkpoint": FINAL_CHECKPOINT_NAME,
        "best_epoch": best_epoch,
        "best_train_loss_trainmode": float(best_train_loss),
        "best_eval_loss_evalmode": float(best_eval_loss),
        "final_eval_loss_evalmode": float(final_loss),
        "bn_recalibrated": True,
        "eval_drop_last": False,
        "stage2_train_samples": int(len(train_idx)),
        "stage2_val_samples": int(len(val_idx)),
        "stage2_val_ratio": 0.05,
        "ema_used": True,
        "ema_decay": float(ema_decay),
        "augment_flip": False,  # 오프라인 데이터셋 4× 증강 사용 (학습 시 추가 flip 없음)
        "best_train_loss": float(best_train_loss),
        "final_train_loss": float(final_loss),
    }
    return history, final_info


# ──────────────────────────────────────────────────────────────────────────────
# Artifact 저장
# ──────────────────────────────────────────────────────────────────────────────
def save_training_artifacts(output_dir, frequencies, metadata, fold_results, history, final_info, args, runtime_info):
    with open(output_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # 변경: val_ema 컬럼 추가
        writer.writerow(["epoch", "train_loss", "val_loss", "val_ema", "lr"])
        for epoch, tr, vl, ve, lr in zip(
            history["epoch"], history["train_loss"], history["val_loss"], history["val_ema"], history["lr"],
        ):
            writer.writerow([epoch, f"{tr:.8f}", f"{vl:.8f}", f"{ve:.8f}", f"{lr:.8f}"])

    config = {
        "schema_version": 1,
        "model_type": "forward_surrogate",
        "backbone": "resnet50",
        "grid_size": GRID_SIZE,
        "input_size": [GRID_SIZE, GRID_SIZE],
        "output_dim": metadata["freq_points"] * len(DB_KEYS),
        "output_format": "db",
        "target_keys": DB_KEYS,
        "freq_points": metadata["freq_points"],
        "freq_range_ghz": metadata["freq_range_ghz"],
        "frequencies_file": "frequencies.npy",
        "normalization_type": "per_key_minmax",
        "normalization": metadata["normalization"],
        "k_folds": K_FOLDS,
        "accumulation_steps": runtime_info["accum_steps"],
        "batch_size": runtime_info["batch_size"],
        "batch_size_source": runtime_info["batch_size_source"],
        "num_workers": runtime_info["num_workers"],
        "max_epochs_kfold": args.max_epochs,
        "max_lr": args.max_lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "max_walltime_hours": args.max_walltime_hours,
        "pretrained": args.pretrained,
        "channel_loss_weights": CHANNEL_LOSS_WEIGHTS,
        "smart_training": {
            "ema_used": True,
            "ema_decay": args.ema_decay,
            "augment_flip": False,  # 오프라인 데이터셋 4× 증강 (x/y/원점 대칭) 사용
            "offline_dataset_augmentation": "x_sym + y_sym + origin_sym (4x)",
            "scheduler": "CosineAnnealingWarmRestarts(T_0=patience*steps_per_epoch, T_mult=2)",
            "loss": "ChannelWeightedSmoothL1Loss",
            "best_state_selection": "min(val_raw, val_ema) per epoch",
        },
        "h100_optim": {
            "amp_dtype": str(runtime_info["amp_dtype"]).replace("torch.", ""),
            "channels_last": bool(runtime_info["use_channels_last"]),
            "tf32_enabled": bool(runtime_info["tf32_enabled"]),
            "cudnn_benchmark": bool(runtime_info["cudnn_benchmark"]),
            "device_name": runtime_info.get("device_name"),
            "vram_gb": runtime_info.get("vram_gb"),
        },
        "use_ensemble": True,
        "checkpoint_format": f"ensemble/{CHECKPOINT_PATTERN}",
        "ensemble_checkpoints": [result["checkpoint"] for result in fold_results],
        "fold_results": fold_results,
        **final_info,
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(output_dir / "ensemble_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "use_ensemble": True,
            "fold_results": fold_results,
            "ensemble_checkpoints": [result["checkpoint"] for result in fold_results],
            "final_checkpoint": FINAL_CHECKPOINT_NAME,
        }, f, indent=2)

    np.save(output_dir / "frequencies.npy", frequencies)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    default_layouts, default_sparams = default_data_paths()
    parser = argparse.ArgumentParser(
        description="25x25 ResNet50 Forward Model Training — H100 최적화판",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--layouts", type=str, default=str(default_layouts))
    parser.add_argument("--sparams", type=str, default=str(default_sparams))
    parser.add_argument("--output", type=str, default=str(default_output_dir()))
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="수동 batch size (지정 시 자동 감지 무시).",
    )
    parser.add_argument(
        "--auto-batch", action="store_true", default=True,
        help="기본값 ON. VRAM 을 보고 batch_size 를 자동 결정 (H100 80GB → 1024 등).",
    )
    parser.add_argument("--max-lr", type=float, default=DEFAULT_MAX_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-walltime-hours", type=float, default=DEFAULT_MAX_WALLTIME_HOURS,
                        help="Stage 1/2 합계 최대 학습 시간. 기본 9.5h 로 12시간 초과 방지.")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=True,
                        help="ImageNet pretrained 가중치를 사용하지 않음 (기본: 사용)")
    parser.add_argument("--no-channels-last", dest="channels_last", action="store_false", default=True,
                        help="channels_last memory_format 비활성화 (기본: 활성화)")
    parser.add_argument("--force-fp16", action="store_true", default=False,
                        help="Ampere/Hopper GPU 에서도 강제로 fp16 autocast 사용 (기본: bfloat16 선호)")
    parser.add_argument("--ema-decay", type=float, default=DEFAULT_EMA_DECAY,
                        help="EMA decay. 1에 가까울수록 부드러움. ResNet50 기본 0.9995.")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_all_seeds(SEED)
    configure_backend_for_h100()

    print(TITLE)
    print("ResNet50 25x25 Forward Model Training  (H100 최적화)")
    print(TITLE)

    run_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Device 정보 출력 ────────────────────────────────────────────────
    device_name = None
    vram_gb = None
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(device)
        device_name = prop.name
        vram_gb = float(prop.total_memory / 1e9)
        print(f"Device           : {device} ({device_name})")
        print(f"VRAM             : {vram_gb:.1f} GB")
        print(f"Compute capability: {torch.cuda.get_device_capability(device)}")
    else:
        print(f"Device           : {device}")

    # ── 출력 경로 ──────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ensemble").mkdir(parents=True, exist_ok=True)

    # ── Batch size / num_workers / accum / autocast dtype 결정 ─────────
    batch_size = auto_batch_size(device, manual_batch=args.batch_size)
    batch_size_source = "manual" if args.batch_size is not None else "auto"
    accum_steps = resolve_accum_steps(batch_size)
    num_workers = resolve_num_workers(args.num_workers)
    amp_dtype = torch.float16 if args.force_fp16 else pick_autocast_dtype(device)
    use_channels_last = bool(args.channels_last) and device.type == "cuda"

    print(f"Layouts path     : {args.layouts}")
    print(f"S-params path    : {args.sparams}")
    print(f"Output dir       : {output_dir}")
    print(f"Grid size        : {GRID_SIZE}x{GRID_SIZE}")
    print(f"Batch size       : {batch_size}  ({batch_size_source})  "
          f"(effective {batch_size * accum_steps})")
    print(f"Accum steps      : {accum_steps}")
    print(f"Max epochs (CV)  : {args.max_epochs}")
    print(f"Max LR           : {args.max_lr}")
    print(f"Patience         : {args.patience}")
    print(f"Num workers      : {num_workers}")
    print(f"Max walltime     : {args.max_walltime_hours:.2f} hours")
    print(f"Loss             : ChannelWeightedSmoothL1Loss")
    print(f"  weights        : {CHANNEL_LOSS_WEIGHTS}")
    print(f"Optimizer        : AdamW (weight_decay={args.weight_decay})")
    print(f"Scheduler        : CosineAnnealingWarmRestarts(T_0={args.patience}*steps, T_mult=2)")
    print(f"Backbone         : ResNet50 (Bottleneck, 2048-dim)")
    print(f"AMP dtype        : {amp_dtype}")
    print(f"channels_last    : {use_channels_last}")
    print(f"TF32 / cudnn.benchmark : enabled (configure_backend_for_h100)")
    print(f"Augment (flip)   : disabled (dataset already 4x offline-augmented)")
    print(f"EMA decay        : {args.ema_decay}")

    # ── 데이터 로딩 ────────────────────────────────────────────────────
    X, y, frequencies, metadata = load_training_data(Path(args.layouts), Path(args.sparams))
    print(SEP)
    print(f"Loaded samples   : {len(X):,}")
    print(f"Target dim       : {y.shape[1]}")
    print(f"Frequency points : {metadata['freq_points']}")

    # ── Stage 1 ───────────────────────────────────────────────────────
    optimal_epochs, fold_results = kfold_find_optimal_epochs(
        X, y, device, output_dir,
        max_epochs=args.max_epochs,
        batch_size=batch_size,
        max_lr=args.max_lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        pretrained=args.pretrained,
        num_workers=num_workers,
        start_time=run_start,
        max_walltime_hours=args.max_walltime_hours,
        amp_dtype=amp_dtype,
        use_channels_last=use_channels_last,
        ema_decay=args.ema_decay,
    )

    # ── Stage 2 ───────────────────────────────────────────────────────
    history, final_info = final_train(
        X, y, device, output_dir,
        optimal_epochs=optimal_epochs,
        batch_size=batch_size,
        max_lr=args.max_lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        pretrained=args.pretrained,
        num_workers=num_workers,
        start_time=run_start,
        max_walltime_hours=args.max_walltime_hours,
        amp_dtype=amp_dtype,
        use_channels_last=use_channels_last,
        ema_decay=args.ema_decay,
    )

    runtime_info = {
        "batch_size": batch_size,
        "batch_size_source": batch_size_source,
        "accum_steps": accum_steps,
        "num_workers": num_workers,
        "amp_dtype": amp_dtype,
        "use_channels_last": use_channels_last,
        "tf32_enabled": True,
        "cudnn_benchmark": True,
        "device_name": device_name,
        "vram_gb": vram_gb,
    }
    save_training_artifacts(output_dir, frequencies, metadata, fold_results, history, final_info, args, runtime_info)

    print(TITLE)
    print("Training complete (ResNet50, H100 최적화)")
    print(f"Final checkpoint : {output_dir / FINAL_CHECKPOINT_NAME}")
    print(f"Total runtime    : {elapsed_hours(run_start):.2f} hours")
    print(TITLE)


if __name__ == "__main__":
    main()
