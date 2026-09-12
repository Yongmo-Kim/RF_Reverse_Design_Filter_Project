#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_test.py: Forward Model Training Pipeline  (v3 — H100 최적화)
===================================================================
[스펙 기준]
  - 기판: 20.0×20.0 mm | 패턴 영역 마진: 5.0 mm
  - 픽셀 크기: 0.4 mm | 그리드: 25×25 (625 pixels)
  - 포트: Width=1.2mm, Length=3.0mm, Y=10.0mm (기판 중심)

[주파수 해석 구간 — 총 91 pts]
  0.1 GHz (1 pt)
  0.5 ~ 1.5 GHz  @ 0.5 GHz 간격  ( 3 pts)
  2.0 ~ 12.0 GHz @ 0.2 GHz 간격  (51 pts)
  12.5 ~ 30.0 GHz @ 0.5 GHz 간격 (36 pts)
  → output_dim = 91 × 3 = 273  (S11_dB / S21_dB / S22_dB)

[데이터 형식] npz + s2p 지원
  - 단일 병합: layouts.npz (key='layouts') + s_params.npz (keys='s11_db','s21_db','s22_db')
  - 개별 쌍  : <dir>/*.npz (layout) + 동명 *.s2p (S-param) 페어

[v3 업그레이드 요약]
  1. stem_type 옵션 추가
       'standard' : 7×7 Conv(stride=1) + Identity pool (기존)
       'small'    : 3×3 Conv × 2 (stride=1), pool 완전 제거 → 25×25 최적화
  2. --no-pretrained / --pretrained-path 옵션
       오프라인 환경 로컬 가중치 경로 지정 또는 랜덤 초기화 강제
  3. --norm-type 옵션
       'minmax'   : 채널별 Min-Max [0,1] 정규화 (기존)
       'standard' : 채널별 Z-score → clip(±3σ) → [0,1] 스케일링
                    S21(-110~0) 처럼 채널 간 스케일 차이가 큰 경우 권장
  4. --auto-batch / --batch-size 옵션
       H100 100 GB VRAM 자동 감지 → 최적 batch size 제안/적용
       수동으로 --batch-size 1024 등 대형 배치 설정 가능
  5. --num-workers / CosineAnnealing + Warmup 스케줄러 개선
  6. Stage-2 BO/DE 검증 포함 (scikit-learn / scipy 필요)
"""
import os, sys, copy, json, time, argparse
import math
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from torchvision.models import densenet121, DenseNet121_Weights
from models import DenseNet25x25 as SharedDenseNet25x25

# ── 선택적 의존성 (Stage-2 BO/DE 에서 사용) ──────────────────────────────────
try:
    from sklearn.ensemble import RandomForestRegressor as _RF
    from scipy.stats import norm as _sp_norm
    _BO_AVAILABLE = True
except ImportError:
    _BO_AVAILABLE = False

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
TITLE = '=' * 70
SEP   = '-' * 70

# ─────────────────────────────────────────────────────────────────────────────
# [스펙] 그리드 및 주파수 상수
# ─────────────────────────────────────────────────────────────────────────────
GRID_SIZE = 25   # 25×25 픽셀 (0.4mm × 25 = 10mm 패턴 영역)

# 91-point 주파수 배열 (GHz)
TARGET_FREQS = np.concatenate([
    np.array([0.1]),                         # 1 pt
    np.arange(0.5,  1.6,  0.5),             # 3 pts : 0.5, 1.0, 1.5
    np.arange(2.0,  12.1, 0.2),             # 51 pts: 2.0~12.0 @0.2
    np.arange(12.5, 30.1, 0.5),             # 36 pts: 12.5~30.0 @0.5
]).astype(np.float32)                        # 총 91 pts

N_FREQS    = len(TARGET_FREQS)               # 91
N_CHANNELS = 8                               # 변경: S11/S12/S21/S22 의 dB + degree 8채널로 확장
OUTPUT_DIM = N_FREQS * N_CHANNELS            # 변경: 91 * 8 = 728

DB_KEYS          = ['s11_db', 's12_db', 's21_db', 's22_db']  # 변경: magnitude 채널 4개
DEG_KEYS         = ['s11_deg', 's12_deg', 's21_deg', 's22_deg']  # 변경: phase 채널 4개
TARGET_KEYS      = DB_KEYS + DEG_KEYS  # 변경: 전체 출력 채널 순서 정의
DB_KEYS = ['s11_db', 's12_db', 's21_db', 's22_db']  # 변경: 깨진 병합 라인과 무관하게 magnitude 채널 4개를 다시 선언
DEG_KEYS = ['s11_deg', 's12_deg', 's21_deg', 's22_deg']  # 변경: 깨진 병합 라인과 무관하게 phase 채널 4개를 다시 선언
TARGET_KEYS = DB_KEYS + DEG_KEYS  # 변경: 전체 출력 채널 순서를 다시 선언
COMMON_CONFIG_NAME    = 'model_config.json'
LEGACY_CONFIG_NAME    = 'pipeline_config.json'
CHECKPOINT_TEMPLATE   = 'forward_fold{fold}.pt'


def _save_state_dict_atomic(state_dict, path: Path):
    """NFS 등에서 저장 도중 깨짐 방지: 임시 파일 후 rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(state_dict, tmp)
    os.replace(str(tmp), str(path))


def _expected_fold_checkpoints(ensemble_dir: Path, kfolds: int):
    """forward_fold0.pt … forward_fold{k-1}.pt 경로 목록 (순서 고정)."""
    return [ensemble_dir / CHECKPOINT_TEMPLATE.format(fold=i) for i in range(kfolds)]


def _verify_fold_checkpoints(ensemble_dir: Path, kfolds: int, min_bytes=64):
    """Stage-1 직후 디스크에 k개 파일이 모두 존재하는지 확인."""
    bad = []
    for ck in _expected_fold_checkpoints(ensemble_dir, kfolds):
        if not ck.is_file() or ck.stat().st_size < min_bytes:
            bad.append(ck)
    return bad


def _channel_arrays_to_map(*arrays):
    """채널 배열을 TARGET_KEYS 순서의 dict로 변환."""  # 변경: 8채널 처리를 위한 공통 helper 추가
    return {key: arr for key, arr in zip(TARGET_KEYS, arrays)}


def _split_raw_target_map(y_raw, n_freqs):
    """(N, output_dim) raw target를 TARGET_KEYS dict로 분리."""  # 변경: 3채널 하드코딩 슬라이싱 제거
    return {
        key: y_raw[:, i * n_freqs:(i + 1) * n_freqs]
        for i, key in enumerate(TARGET_KEYS)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Forward Model (DenseNet121) — inverse_test.py 완전 동기화
# ─────────────────────────────────────────────────────────────────────────────
class DenseNet25x25(nn.Module):
    """DenseNet121 기반 Forward 모델.

    입력 : (B, 1, 25, 25) 바이너리 레이아웃
    출력 : (B, output_dim) 정규화된 S-파라미터 [0, 1]

    stem_type 옵션
    ──────────────
    'standard'  7×7 Conv(stride=1) + Identity pool
                ImageNet 사전학습 가중치 활용 가능. 현행 기본값.
    'small'     3×3 Conv(1→32) → BN → ReLU →
                3×3 Conv(32→64) → BN → ReLU (pool 완전 제거)
                25×25 소형 입력에 최적. 공간 정보를 최대한 보존.
                사전학습 가중치 미지원(랜덤 초기화 자동 적용).
    """

    def __init__(self, output_dim=OUTPUT_DIM,
                 use_pretrained=True,
                 pretrained_path=None,
                 stem_type='standard',
                 freeze_backbone=False):
        super().__init__()
        self.stem_type = stem_type

        # ── 백본 생성 ──────────────────────────────────────────────
        # 'small' stem은 채널 수가 달라 ImageNet 가중치와 호환 불가
        load_imagenet = use_pretrained and (stem_type == 'standard') and (pretrained_path is None)

        if load_imagenet:
            try:
                self.densenet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
                print("  [Backbone] ImageNet pretrained 로드 완료")
            except Exception as e:
                self.densenet = densenet121(weights=None)
                print(f"  [Backbone] WARNING: ImageNet 로드 실패({e}) → Random Init")
        else:
            self.densenet = densenet121(weights=None)
            if pretrained_path:
                print(f"  [Backbone] 로컬 가중치 로드: {pretrained_path}")
            else:
                reason = "'small' stem" if stem_type == 'small' else "no-pretrained"
                print(f"  [Backbone] Random Init ({reason})")

        if freeze_backbone:
            for p in self.densenet.parameters():
                p.requires_grad = False

        # ── Stem 교체 ──────────────────────────────────────────────
        if stem_type == 'small':
            # ── small stem: 3×3 두 단계, 다운샘플 없음 ─────────────
            # conv0 슬롯: 1 → 32  (3×3, stride=1)
            self.densenet.features.conv0 = nn.Conv2d(1, 32, 3, 1, 1, bias=False)
            self.densenet.features.norm0 = nn.BatchNorm2d(32)
            # relu0 는 그대로 유지
            # pool0 슬롯: 32 → 64 Conv + BN + ReLU (최종 출력채널=64 유지)
            self.densenet.features.pool0 = nn.Sequential(
                nn.Conv2d(32, 64, 3, 1, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            print(f"  [Stem] 'small': 3×3 Conv×2, no pool  (25×25 최적)")
        else:
            # ── standard stem: 7×7 stride-1 + Identity pool ─────────
            old_conv = self.densenet.features.conv0          # 3→64, 7×7
            new_conv = nn.Conv2d(1, 64, 7, 1, 3, bias=False)
            if load_imagenet:
                with torch.no_grad():
                    new_conv.weight = nn.Parameter(
                        old_conv.weight.mean(dim=1, keepdim=True))
            self.densenet.features.conv0 = new_conv
            self.densenet.features.pool0 = nn.Identity()
            for p in self.densenet.features.conv0.parameters():
                p.requires_grad = True
            print(f"  [Stem] 'standard': 7×7 stride-1, Identity pool")

        # ── 로컬 가중치 덮어쓰기 (stem 교체 후 적용) ───────────────
        if pretrained_path:
            try:
                try:
                    ck = torch.load(pretrained_path, map_location='cpu', weights_only=True)
                except Exception:
                    ck = torch.load(pretrained_path, map_location='cpu', weights_only=False)
                # state_dict 또는 {'model':...} 형식 모두 처리
                sd = ck.get('model', ck.get('state_dict', ck))
                miss, unexp = self.densenet.load_state_dict(sd, strict=False)
                print(f"  [Pretrained] 로컬 로드 완료  miss={len(miss)}  unexp={len(unexp)}")
            except Exception as e:
                print(f"  [Pretrained] WARNING: 로컬 로드 실패({e}) → 현재 가중치 유지")

        # ── Classifier 교체 ────────────────────────────────────────
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(True),
            nn.Linear(256, output_dim), nn.Sigmoid())
        self._init_fc()

    def _init_fc(self):
        for m in self.densenet.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        f = self.densenet.features(x)
        o = torch.nn.functional.relu(f, inplace=True)
        o = torch.nn.functional.adaptive_avg_pool2d(o, (1, 1))
        return self.densenet.classifier(torch.flatten(o, 1))


DenseNet25x25 = SharedDenseNet25x25


# ─────────────────────────────────────────────────────────────────────────────
# S2P 파서
# ─────────────────────────────────────────────────────────────────────────────
def parse_s2p(filepath, target_freqs_ghz=None):
    """Touchstone .s2p 파일 파싱 → (freqs_GHz, s11/s12/s21/s22의 dB+degree).

    포맷 자동 감지: RI (실수/허수), MA (크기/각도), DB (dB/각도).
    target_freqs_ghz 지정 시 해당 주파수로 선형 보간.
    """
    freqs, rows = [], []
    fmt         = 'RI'
    freq_scale  = 1e9   # 기본 GHz

    with open(filepath, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('!'): continue
            if line.startswith('#'):
                parts = line.upper().split()
                if len(parts) >= 2:
                    freq_scale = {'GHZ': 1e9, 'MHZ': 1e6,
                                  'KHZ': 1e3,  'HZ':  1.0}.get(parts[1], 1e9)
                if len(parts) >= 4:
                    fmt = parts[3]  # RI / MA / DB
                continue
            vals = list(map(float, line.split()))
            if len(vals) >= 9:
                freqs.append(vals[0] * freq_scale / 1e9)   # → GHz
                rows.append(vals[1:9])

    freqs = np.array(freqs, dtype=np.float64)
    data  = np.array(rows,  dtype=np.float64)

    if fmt == 'RI':
        s11 = data[:, 0] + 1j * data[:, 1]
        s21 = data[:, 2] + 1j * data[:, 3]
        s12 = data[:, 4] + 1j * data[:, 5]
        s22 = data[:, 6] + 1j * data[:, 7]
        s11_db = 20 * np.log10(np.abs(s11) + 1e-12)
        s12_db = 20 * np.log10(np.abs(s12) + 1e-12)
        s21_db = 20 * np.log10(np.abs(s21) + 1e-12)
        s22_db = 20 * np.log10(np.abs(s22) + 1e-12)
        s11_deg = np.degrees(np.angle(s11))
        s12_deg = np.degrees(np.angle(s12))
        s21_deg = np.degrees(np.angle(s21))
        s22_deg = np.degrees(np.angle(s22))
    elif fmt == 'MA':
        s11_db = 20 * np.log10(data[:, 0] + 1e-12)
        s21_db = 20 * np.log10(data[:, 2] + 1e-12)
        s12_db = 20 * np.log10(data[:, 4] + 1e-12)
        s22_db = 20 * np.log10(data[:, 6] + 1e-12)
        s11_deg = data[:, 1]
        s21_deg = data[:, 3]
        s12_deg = data[:, 5]
        s22_deg = data[:, 7]
    else:   # DB
        s11_db = data[:, 0]
        s21_db = data[:, 2]
        s12_db = data[:, 4]
        s22_db = data[:, 6]
        s11_deg = data[:, 1]
        s21_deg = data[:, 3]
        s12_deg = data[:, 5]
        s22_deg = data[:, 7]

    arrays = [s11_db, s12_db, s21_db, s22_db, s11_deg, s12_deg, s21_deg, s22_deg]  # 변경: 8채널 모두 유지
    if target_freqs_ghz is not None:
        arrays = [np.interp(target_freqs_ghz, freqs, arr).astype(np.float32) for arr in arrays]  # 변경: 모든 채널에 동일 보간 적용
        out_freqs = target_freqs_ghz.astype(np.float32)
    else:
        arrays = [arr.astype(np.float32) for arr in arrays]
        out_freqs = freqs.astype(np.float32)

    return (out_freqs, *arrays)


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 로더
# ─────────────────────────────────────────────────────────────────────────────
def _load_merged_npz(layouts_file, sparams_file):
    """layouts.npz + s_params.npz (단일 병합 파일 로딩)."""
    L_data = np.load(layouts_file, allow_pickle=True)
    S_data = np.load(sparams_file, allow_pickle=True)

    # 요청된 키 값으로 정확히 접근
    layouts = L_data['layouts'].astype(np.float32)
    s11 = S_data['s11_db'].astype(np.float32)
    s21 = S_data['s21_db'].astype(np.float32)
    s22 = S_data['s22_db'].astype(np.float32)

    # 주파수 로드 (있으면 사용, 없으면 TARGET_FREQS 적용)
    if 'frequencies' in S_data.files:
        src_freqs = S_data['frequencies'].astype(np.float32)
    else:
        src_freqs = TARGET_FREQS
        print("  ⚠ frequencies 키 없음 — TARGET_FREQS 91pts 사용")

    # 주파수 수 확인 → 필요 시 보간
    if s11.shape[1] != N_FREQS:
        print(f"  주파수 보간: {s11.shape[1]} pts → {N_FREQS} pts")
        interp = lambda arr: np.stack([
            np.interp(TARGET_FREQS, src_freqs, arr[i]) for i in range(len(arr))
        ]).astype(np.float32)
        s11 = interp(s11)
        s21 = interp(s21)
        s22 = interp(s22)

    return layouts, s11, s21, s22, TARGET_FREQS


def _load_from_dir(data_dir):
    """개별 파일 형식: <dir>/*.npz (layout) + 동명 *.s2p 페어."""
    data_dir = Path(data_dir)
    npz_files = sorted(data_dir.glob('*.npz'))
    if not npz_files:
        raise FileNotFoundError(f"npz 파일이 없습니다: {data_dir}")

    layouts_list, s11_list, s21_list, s22_list = [], [], [], []
    skipped = 0

    for npz_path in npz_files:
        s2p_path = npz_path.with_suffix('.s2p')
        if not s2p_path.exists():
            skipped += 1
            continue

        # layout 로드
        d = np.load(npz_path, allow_pickle=True)
        key = next((k for k in ['layout', 'layouts', 'data'] if k in d.files),
                   d.files[0])
        layout = d[key].astype(np.float32)
        if layout.ndim == 1:
            side = int(np.round(np.sqrt(len(layout))))
            layout = layout.reshape(side, side)

        # s2p 파싱 (TARGET_FREQS로 보간)
        _, s11, s21, s22 = parse_s2p(s2p_path, TARGET_FREQS)

        layouts_list.append(layout)
        s11_list.append(s11)
        s21_list.append(s21)
        s22_list.append(s22)

    if skipped:
        print(f"  ⚠ s2p 없는 npz {skipped}개 스킵")

    return (np.stack(layouts_list).astype(np.float32),
            np.stack(s11_list).astype(np.float32),
            np.stack(s21_list).astype(np.float32),
            np.stack(s22_list).astype(np.float32),
            TARGET_FREQS)


def normalize_sparams(s11, s21, s22, norm_type='minmax', clip_sigma=3.0):
    """
    S-파라미터 정규화 → (s_norm, norm_params)

    norm_type
    ─────────
    'minmax'   : 채널별 Min-Max → [0, 1]
                 간단하고 역정규화가 명확. 이상치에 민감.

    'standard' : 채널별 Z-score → clip(±clip_sigma) → [0, 1]
                 S21: -110 ~ 0 처럼 채널 간 스케일 차가 클 때 권장.
                 신경망 학습 안정성 향상. clip_sigma=3.0 기본.

    반환 norm_params 포맷 (inverse_test.py denorm_clamp 호환)
    ──────────────────────────────────────────────────────────
    minmax   : {'s11_db': {'type':'minmax', 'min':v, 'max':v}, ...}
    standard : {'s11_db': {'type':'standard', 'mean':v, 'std':v,
                           'clip':clip_sigma}, ...}
    """
    arrays, norm_params = [], {}

    for key, arr in zip(DB_KEYS, [s11, s21, s22]):
        if norm_type == 'standard':
            mean_ = float(arr.mean())
            std_  = float(arr.std()) + 1e-8
            z     = (arr - mean_) / std_
            z_clp = np.clip(z, -clip_sigma, clip_sigma)
            # → [0, 1]
            normed = (z_clp + clip_sigma) / (2.0 * clip_sigma)
            norm_params[key] = {
                'type':  'standard',
                'mean':  mean_,
                'std':   std_,
                'clip':  clip_sigma,
                # 원시 데이터 범위(참고용)
                'raw_min': float(arr.min()),
                'raw_max': float(arr.max()),
            }
        else:  # minmax
            vmin = float(arr.min())
            vmax = float(arr.max())
            normed = (arr - vmin) / (vmax - vmin + 1e-8)
            norm_params[key] = {'type': 'minmax', 'min': vmin, 'max': vmax}

        arrays.append(normed.astype(np.float32))

    s_norm = np.concatenate(arrays, axis=1)   # (N, 3×91)
    return s_norm, norm_params


def apply_norm_params(s11, s21, s22, norm_params):
    """주어진 norm_params를 사용해 S-파라미터를 정규화."""
    arrays = []
    for key, arr in zip(DB_KEYS, [s11, s21, s22]):
        p = norm_params[key]
        if p.get('type', 'minmax') == 'standard':
            clip_sigma = float(p.get('clip', 3.0))
            z = (arr - p['mean']) / (p['std'] + 1e-8)
            z = np.clip(z, -clip_sigma, clip_sigma)
            normed = (z + clip_sigma) / (2.0 * clip_sigma)
        else:
            lo, hi = p['min'], p['max']
            normed = (arr - lo) / (hi - lo + 1e-8)
        arrays.append(normed.astype(np.float32))
    return np.concatenate(arrays, axis=1).astype(np.float32)


def load_data(layouts_path, sparams_path, norm_type='minmax'):
    """
    데이터 로드 → (X_tensor, y_tensor, freqs, norm_params, target_keys).

    layouts_path 가 디렉토리이면 개별 npz+s2p 모드,
    파일이면 기존 단일 병합 모드.
    norm_type : 'minmax' | 'standard'
    """
    lp = Path(layouts_path)
    sp = Path(sparams_path)

    if lp.is_dir():
        print(f"  [개별 파일 모드] 디렉토리: {lp}")
        layouts, s11, s21, s22, freqs = _load_from_dir(lp)
    else:
        print(f"  [병합 파일 모드] {lp.name} + {sp.name}")
        layouts, s11, s21, s22, freqs = _load_merged_npz(lp, sp)

    # 레이아웃 크기 검증
    N = len(layouts)
    if layouts.shape[1:] != (GRID_SIZE, GRID_SIZE):
        print(f"  ⚠ 레이아웃 크기 {layouts.shape[1:]} ≠ ({GRID_SIZE},{GRID_SIZE}) "
              f"— 리사이즈 시도")
        import torch.nn.functional as F_
        t = torch.FloatTensor(layouts).unsqueeze(1)
        t = F_.interpolate(t, size=(GRID_SIZE, GRID_SIZE), mode='nearest')
        layouts = t.squeeze(1).numpy()

    # 샘플 수 일치 검증
    for arr, name in [(s11, 's11_db'), (s21, 's21_db'), (s22, 's22_db')]:
        if len(arr) != N:
            raise ValueError(f"샘플 수 불일치: layouts={N}, {name}={len(arr)}")

    # 추론 기준(reference) 정규화는 전체 데이터로 저장하되,
    # K-fold 학습/검증 손실은 fold-train 기준 정규화를 stage1에서 별도로 적용한다.
    s_norm_ref, norm_params = normalize_sparams(s11, s21, s22, norm_type=norm_type)

    layouts_bin = (layouts > 0.5).astype(np.float32)

    print(f"  샘플 수   : {N}")
    print(f"  레이아웃  : {layouts_bin.shape}  ({GRID_SIZE}×{GRID_SIZE} grid)")
    print(f"  주파수    : {freqs[0]:.2f}~{freqs[-1]:.2f} GHz  ({len(freqs)} pts)")
    print(f"  정규화    : {norm_type}")
    print(f"  출력 채널 : {DB_KEYS}  → output_dim={s_norm_ref.shape[1]}")
    print("  CV 정규화 : fold-train 통계 사용 (val 누수 완화)")
    for k in DB_KEYS:
        p = norm_params[k]
        if p['type'] == 'minmax':
            print(f"    {k}: [{p['min']:.1f}, {p['max']:.1f}] dB  (minmax)")
        else:
            print(f"    {k}: mean={p['mean']:.1f} std={p['std']:.1f} dB  "
                  f"clip=±{p['clip']}σ  (standard)")

    X = torch.FloatTensor(layouts_bin).unsqueeze(1)
    y_raw = np.concatenate([s11, s21, s22], axis=1).astype(np.float32)
    return X, y_raw, freqs, norm_params, DB_KEYS


def _load_merged_npz(layouts_file, sparams_file):
    """변경: layouts.npz + s_params.npz에서 8채널(dB+degree) 전체 로드."""
    L_data = np.load(layouts_file, allow_pickle=True)
    S_data = np.load(sparams_file, allow_pickle=True)
    layouts = L_data['layouts'].astype(np.float32)
    channel_map = {key: S_data[key].astype(np.float32) for key in TARGET_KEYS}  # 변경: 8채널 전체 로드

    if 'frequencies' in S_data.files:
        src_freqs = S_data['frequencies'].astype(np.float32)
    else:
        src_freqs = TARGET_FREQS
        print("  [Info] frequencies가 없어 TARGET_FREQS 91pts 사용")

    if channel_map['s11_db'].shape[1] != N_FREQS:
        print(f"  주파수 보간: {channel_map['s11_db'].shape[1]} pts -> {N_FREQS} pts")
        interp = lambda arr: np.stack([np.interp(TARGET_FREQS, src_freqs, arr[i]) for i in range(len(arr))]).astype(np.float32)
        for key in TARGET_KEYS:
            channel_map[key] = interp(channel_map[key])  # 변경: 모든 채널에 동일 보간 적용

    return (
        layouts,
        channel_map['s11_db'], channel_map['s12_db'], channel_map['s21_db'], channel_map['s22_db'],
        channel_map['s11_deg'], channel_map['s12_deg'], channel_map['s21_deg'], channel_map['s22_deg'],
        TARGET_FREQS,
    )


def _load_from_dir(data_dir):
    """변경: <dir>/*.npz + 동명 *.s2p 를 8채널(dB+degree)로 로드."""
    data_dir = Path(data_dir)
    npz_files = sorted(data_dir.glob('*.npz'))
    if not npz_files:
        raise FileNotFoundError(f"npz 파일이 없습니다: {data_dir}")

    layouts_list = []
    channel_lists = {key: [] for key in TARGET_KEYS}
    skipped = 0

    for npz_path in npz_files:
        s2p_path = npz_path.with_suffix('.s2p')
        if not s2p_path.exists():
            skipped += 1
            continue
        d = np.load(npz_path, allow_pickle=True)
        key = next((k for k in ['layout', 'layouts', 'data'] if k in d.files), d.files[0])
        layout = d[key].astype(np.float32)
        if layout.ndim == 1:
            side = int(np.round(np.sqrt(len(layout))))
            layout = layout.reshape(side, side)

        _, *channels = parse_s2p(s2p_path, TARGET_FREQS)  # 변경: s2p에서 8채널 전체 파싱
        layouts_list.append(layout)
        for key, arr in zip(TARGET_KEYS, channels):
            channel_lists[key].append(arr)

    if skipped:
        print(f"  [Warn] s2p 없는 npz {skipped}개 스킵")

    return (
        np.stack(layouts_list).astype(np.float32),
        *(np.stack(channel_lists[key]).astype(np.float32) for key in TARGET_KEYS),
        TARGET_FREQS,
    )


def normalize_sparams(channel_map, norm_type='minmax', clip_sigma=3.0):
    """변경: TARGET_KEYS 전체를 채널별로 정규화."""
    arrays, norm_params = [], {}
    for key in TARGET_KEYS:
        arr = channel_map[key]
        if norm_type == 'standard':
            mean_ = float(arr.mean())
            std_ = float(arr.std()) + 1e-8
            z = (arr - mean_) / std_
            z_clp = np.clip(z, -clip_sigma, clip_sigma)
            normed = (z_clp + clip_sigma) / (2.0 * clip_sigma)
            norm_params[key] = {
                'type': 'standard', 'mean': mean_, 'std': std_, 'clip': clip_sigma,
                'raw_min': float(arr.min()), 'raw_max': float(arr.max()),
            }
        else:
            vmin = float(arr.min())
            vmax = float(arr.max())
            normed = (arr - vmin) / (vmax - vmin + 1e-8)
            norm_params[key] = {'type': 'minmax', 'min': vmin, 'max': vmax}
        arrays.append(normed.astype(np.float32))
    s_norm = np.concatenate(arrays, axis=1)
    return s_norm, norm_params


def apply_norm_params(channel_map, norm_params):
    """변경: TARGET_KEYS 전체에 동일 정규화 파라미터 적용."""
    arrays = []
    for key in TARGET_KEYS:
        arr = channel_map[key]
        p = norm_params[key]
        if p.get('type', 'minmax') == 'standard':
            clip_sigma = float(p.get('clip', 3.0))
            z = (arr - p['mean']) / (p['std'] + 1e-8)
            z = np.clip(z, -clip_sigma, clip_sigma)
            normed = (z + clip_sigma) / (2.0 * clip_sigma)
        else:
            lo, hi = p['min'], p['max']
            normed = (arr - lo) / (hi - lo + 1e-8)
        arrays.append(normed.astype(np.float32))
    return np.concatenate(arrays, axis=1).astype(np.float32)


def load_data(layouts_path, sparams_path, norm_type='minmax'):
    """변경: 데이터 로드 시 dB+degree 8채널 전체를 반환."""
    lp = Path(layouts_path)
    sp = Path(sparams_path)

    if lp.is_dir():
        print(f"  [개별 파일 모드] 디렉터리: {lp}")
        layouts, s11_db, s12_db, s21_db, s22_db, s11_deg, s12_deg, s21_deg, s22_deg, freqs = _load_from_dir(lp)
    else:
        print(f"  [병합 파일 모드] {lp.name} + {sp.name}")
        layouts, s11_db, s12_db, s21_db, s22_db, s11_deg, s12_deg, s21_deg, s22_deg, freqs = _load_merged_npz(lp, sp)

    N = len(layouts)
    if layouts.shape[1:] != (GRID_SIZE, GRID_SIZE):
        print(f"  [Resize] 레이아웃 크기 {layouts.shape[1:]} -> ({GRID_SIZE},{GRID_SIZE})")
        import torch.nn.functional as F_
        t = torch.FloatTensor(layouts).unsqueeze(1)
        t = F_.interpolate(t, size=(GRID_SIZE, GRID_SIZE), mode='nearest')
        layouts = t.squeeze(1).numpy()

    channel_map = _channel_arrays_to_map(s11_db, s12_db, s21_db, s22_db, s11_deg, s12_deg, s21_deg, s22_deg)
    for key in TARGET_KEYS:
        if len(channel_map[key]) != N:
            raise ValueError(f"샘플 수 불일치: layouts={N}, {key}={len(channel_map[key])}")

    s_norm_ref, norm_params = normalize_sparams(channel_map, norm_type=norm_type)
    layouts_bin = (layouts > 0.5).astype(np.float32)

    print(f"  샘플 수  : {N}")
    print(f"  레이아웃  : {layouts_bin.shape}  ({GRID_SIZE}x{GRID_SIZE} grid)")
    print(f"  주파수   : {freqs[0]:.2f}~{freqs[-1]:.2f} GHz  ({len(freqs)} pts)")
    print(f"  정규화   : {norm_type}")
    print(f"  출력 채널 : {TARGET_KEYS}  -> output_dim={s_norm_ref.shape[1]}")

    X = torch.FloatTensor(layouts_bin).unsqueeze(1)
    y_raw = np.concatenate([channel_map[key] for key in TARGET_KEYS], axis=1).astype(np.float32)
    return X, y_raw, freqs, norm_params, TARGET_KEYS


# ─────────────────────────────────────────────────────────────────────────────
# 주파수 가중 손실
# ─────────────────────────────────────────────────────────────────────────────
def make_freq_weight(freqs, n_channels, device,
                     dense_low=2.0, dense_high=12.0,
                     dense_weight=2.0, sparse_weight=1.0):
    """
    주파수별 손실 가중치.
    2~12 GHz (0.2 GHz 고밀도 해석 구간) → 2× 가중치,
    나머지 구간 → 1× 가중치.
    채널 수만큼 반복하여 전체 output_dim 크기의 벡터 반환.
    """
    w = np.where(
        (freqs >= dense_low) & (freqs <= dense_high),
        dense_weight, sparse_weight
    ).astype(np.float32)
    w = w / w.mean()
    return torch.FloatTensor(np.tile(w, n_channels)).to(device)


class FreqWeightedSmoothL1(nn.Module):
    """주파수 가중 SmoothL1 손실."""
    def __init__(self, freq_weight):
        super().__init__()
        self.register_buffer('weight', freq_weight)
        self.sl1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred, target):
        return (self.sl1(pred, target) * self.weight).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 학습 루프
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion,
                    scaler, scheduler, device, accum=2):
    model.train()
    total = 0.0
    optimizer.zero_grad()
    for i, (imgs, tgts) in enumerate(loader):
        imgs, tgts = imgs.to(device), tgts.to(device)
        with autocast(device_type=device.type):
            loss = criterion(model(imgs), tgts) / accum
        scaler.scale(loss).backward()
        if (i + 1) % accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # 미니배치 단위로 학습률 스케줄러 업데이트
            optimizer.zero_grad()
        total += loss.item() * accum
    # 잔여 미니배치(배수로 나누어떨어지지 않는 경우)도 업데이트 반영
    if len(loader) % accum != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
    return total / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for imgs, tgts in loader:
            imgs, tgts = imgs.to(device), tgts.to(device)
            with autocast(device_type=device.type):
                loss = criterion(model(imgs), tgts)
            total += loss.item()
    return total / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# H100 / GPU 자동 Batch Size 탐지
# ─────────────────────────────────────────────────────────────────────────────
def auto_batch_size(device, manual_batch=None):
    """
    VRAM 용량 기반 권장 batch size 반환.
    manual_batch 가 지정되면 그 값을 그대로 반환(자동 무시).

    25×25 입력 + DenseNet121 기준 경험적 테이블:
      ≥ 80 GB (H100 / A100 80G) → 2048
      ≥ 40 GB (A100 40G)        → 1024
      ≥ 20 GB (A6000 / 3090)    →  512
      ≥ 10 GB                   →  256
      ≥  6 GB                   →  128
      CPU / < 6 GB               →   64
    """
    if manual_batch is not None:
        print(f"  [Batch] 수동 설정: {manual_batch}")
        return manual_batch

    if not torch.cuda.is_available():
        bs = 64
        print(f"  [Batch] CPU 모드 → {bs}")
        return bs

    vram_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    if   vram_gb >= 80:  bs = 2048
    elif vram_gb >= 40:  bs = 1024
    elif vram_gb >= 20:  bs = 512
    elif vram_gb >= 10:  bs = 256
    elif vram_gb >=  6:  bs = 128
    else:                bs = 64

    print(f"  [Batch] VRAM {vram_gb:.0f} GB → 자동 batch_size={bs}")
    return bs


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: K-Fold 앙상블 학습
# ─────────────────────────────────────────────────────────────────────────────
def stage1_forward_ensemble(X, y_raw, freqs, device, output_path,
                             max_epochs=200, batch_size=None,
                             lr=3e-4, patience=30, kfolds=5,
                             num_workers=4,
                             norm_type='minmax',
                             stem_type='standard',
                             use_pretrained=True,
                             pretrained_path=None):
    """K-Fold DenseNet25x25 앙상블 학습 및 체크포인트 저장."""
    print(TITLE)
    print(f"STAGE 1: Forward Ensemble (DenseNet121 × {kfolds}-Fold)")
    print(f"  그리드={GRID_SIZE}×{GRID_SIZE}, output_dim={y_raw.shape[1]}, "
          f"freqs={len(freqs)} pts  stem='{stem_type}'")
    print(TITLE)

    n          = len(X)
    output_dim = y_raw.shape[1]
    n_freqs    = y_raw.shape[1] // 3
    n_ch       = N_CHANNELS
    freq_wt    = make_freq_weight(freqs, n_ch, device)
    criterion  = FreqWeightedSmoothL1(freq_wt)

    ensemble_dir = output_path / 'ensemble'
    os.makedirs(ensemble_dir, exist_ok=True)

    # DataLoader num_workers: Windows는 0 고정
    nw = 0 if os.name == 'nt' else min(num_workers, os.cpu_count() or 4)

    np.random.seed(42)
    idx       = np.random.permutation(n)
    fold_size = n // kfolds
    results   = []

    for fold in range(kfolds):
        vs = fold * fold_size
        ve = (fold + 1) * fold_size if fold < kfolds - 1 else n
        vi = idx[vs:ve]
        ti = np.concatenate([idx[:vs], idx[ve:]])
        accum_steps = 1 if (batch_size is not None and batch_size >= 1024) else 2
        print(f"\n  Fold {fold+1}/{kfolds}: Train={len(ti)}, Val={len(vi)}"
              f"  batch={batch_size}  nw={nw}  accum={accum_steps}")

        # fold-train 통계로 정규화 파라미터 생성 → train/val 모두 동일 파라미터 적용
        train_map = {key: value[ti] for key, value in _split_raw_target_map(y_raw, n_freqs).items()}  # 변경: train raw target를 8채널 dict로 분리
        val_map = {key: value[vi] for key, value in _split_raw_target_map(y_raw, n_freqs).items()}  # 변경: val raw target를 8채널 dict로 분리
        _, fold_norm = normalize_sparams(train_map, norm_type=norm_type)  # 변경: fold 정규화도 8채널 전체 기준
        y_train = apply_norm_params(train_map, fold_norm)  # 변경: train 전체 채널 정규화 적용
        y_val = apply_norm_params(val_map, fold_norm)  # 변경: val 전체 채널 정규화 적용

        tl = DataLoader(TensorDataset(X[ti], torch.FloatTensor(y_train)),
                        batch_size=batch_size, shuffle=True,
                        num_workers=nw, pin_memory=(device.type == 'cuda'),
                        persistent_workers=(nw > 0))
        vl = DataLoader(TensorDataset(X[vi], torch.FloatTensor(y_val)),
                        batch_size=batch_size * 2, shuffle=False,
                        num_workers=nw, pin_memory=(device.type == 'cuda'),
                        persistent_workers=(nw > 0))

        model = DenseNet25x25(
            output_dim=output_dim,
            use_pretrained=use_pretrained,
            pretrained_path=pretrained_path,
            stem_type=stem_type,
        ).to(device)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=1e-2)

        # OneCycleLR 스텝 수는 optimizer.step 횟수와 일치시켜야 함
        steps_ep  = max(1, int(math.ceil(len(tl) / accum_steps)))
        scheduler = OneCycleLR(optimizer, max_lr=lr,
                               epochs=max_epochs, steps_per_epoch=steps_ep,
                               pct_start=0.1, anneal_strategy='cos')
        scaler = GradScaler(device.type)

        best_val, best_state, best_ep, no_imp = float('inf'), None, 0, 0

        for ep in range(1, max_epochs + 1):
            train_one_epoch(
                model, tl, optimizer, criterion, scaler, scheduler, device,
                accum=accum_steps
            )
            val = evaluate(model, vl, criterion, device)
            
            if val < best_val - 1e-5:
                best_val, best_ep, no_imp = val, ep, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                no_imp += 1

            if no_imp >= patience:
                print(f"    Early stop ep {ep}")
                break
            if ep % 20 == 0 or ep == 1:
                print(f"    Ep {ep:3d} | Val: {val:.6f} | Best: {best_val:.6f} (ep {best_ep})")

        if best_state:
            model.load_state_dict(best_state)
        ck = ensemble_dir / CHECKPOINT_TEMPLATE.format(fold=fold)
        _save_state_dict_atomic(model.state_dict(), ck)
        results.append({'fold': fold, 'checkpoint': ck.name,
                        'best_epoch': best_ep, 'best_val_loss': float(best_val),
                        'stem_type': stem_type,
                        'norm_params': fold_norm})
        print(f"    Saved: {ck.name}  (ep {best_ep}, val={best_val:.6f})")

        del model, optimizer, scheduler, scaler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    bad_ck = _verify_fold_checkpoints(ensemble_dir, kfolds)
    if bad_ck:
        print("  ❌ ensemble 체크포인트 검증 실패 (없거나 크기 이상):")
        for p in bad_ck:
            ex, sz = p.exists(), (p.stat().st_size if p.exists() else 0)
            print(f"     {p}  exists={ex}  size={sz}")
        extras = sorted(ensemble_dir.glob('*.pt'))
        print(f"  ensemble/*.pt 현재 목록 ({len(extras)}개):")
        for p in extras:
            print(f"     {p.name}  {p.stat().st_size}")
        raise RuntimeError(
            "K-fold 체크포인트가 디스크에 완전하지 않습니다. "
            "쿼터·NFS·출력 경로·동시 실행 여부를 확인하세요.")
    brief = []
    for ck in _expected_fold_checkpoints(ensemble_dir, kfolds):
        brief.append(f"{ck.name}={ck.stat().st_size // 1024}KiB")
    print(f"  [검증] ensemble 체크포인트 {kfolds}개 OK  " + "  ".join(brief))

    # ensemble_meta.json
    meta = {
        'schema_version': 1,
        'model_type':     'forward_surrogate',
        'backbone':       'densenet121',
        'stem_type':      stem_type,
        'checkpoint_format': f'ensemble/{CHECKPOINT_TEMPLATE}',
        'kfolds':    kfolds,
        'output_dim': output_dim,
        'target_keys': TARGET_KEYS,  # 변경: 저장 메타데이터도 8채널 순서 기록
        'fold_results': results,
    }
    for d in [output_path, ensemble_dir]:
        with open(d / 'ensemble_meta.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

    avg_val = np.mean([r['best_val_loss'] for r in results])
    print(f"\n  앙상블 완료: {kfolds}개 모델 저장  (평균 val loss={avg_val:.6f})")
    return output_dim, results



# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: BO / DE 검증 (훈련된 앙상블 surrogate 재활용)
# ─────────────────────────────────────────────────────────────────────────────
def _apply_port_constraints(pop):
    """포트 픽셀 고정 (25×25 기준, rows 11:14, cols 0:2 / 23:25)."""
    pop[:, 11:14, :2]  = 1.0
    pop[:, 11:14, 23:] = 1.0
    return pop


def _make_mosaic_layout(rng, size=GRID_SIZE, fill_lo=0.18, fill_hi=0.68):
    """체커보드 기반 모자이크 랜덤 패턴 1개 생성."""
    yy, xx = np.indices((size, size))
    checker = ((yy + xx) % 2).astype(np.float32)  # 50% fill

    # target fill에 맞게 체커보드에서 랜덤 뒤집기
    target_fill = rng.uniform(fill_lo, fill_hi)
    pat = checker.copy()
    if target_fill > 0.5:
        need = int(round((target_fill - 0.5) * size * size))
        zero_idx = np.argwhere(pat < 0.5)
        if len(zero_idx) > 0 and need > 0:
            pick = zero_idx[rng.choice(len(zero_idx), size=min(need, len(zero_idx)), replace=False)]
            pat[pick[:, 0], pick[:, 1]] = 1.0
    elif target_fill < 0.5:
        need = int(round((0.5 - target_fill) * size * size))
        one_idx = np.argwhere(pat > 0.5)
        if len(one_idx) > 0 and need > 0:
            pick = one_idx[rng.choice(len(one_idx), size=min(need, len(one_idx)), replace=False)]
            pat[pick[:, 0], pick[:, 1]] = 0.0

    # 국소 고정 패턴을 피하기 위한 약한 지터
    jitter_mask = rng.random((size, size)) < 0.06
    pat[jitter_mask] = 1.0 - pat[jitter_mask]
    return pat.astype(np.float32)


def _make_random_pop(n, size=GRID_SIZE, seed=None):
    """모자이크 기반 binary 초기 집단 생성."""
    rng = np.random.default_rng(seed)
    pop = np.stack([_make_mosaic_layout(rng, size=size) for _ in range(n)], axis=0)
    _apply_port_constraints(pop)
    return pop


def _norm_to_db_np(arr, p):
    if p.get('type', 'minmax') == 'standard':
        clip = float(p.get('clip', 3.0))
        z = np.clip(arr, 0.0, 1.0) * (2.0 * clip) - clip
        return z * p['std'] + p['mean']
    lo, hi = p['min'], p['max']
    return np.clip(arr, 0.0, 1.0) * (hi - lo) + lo


def _db_to_norm_np(arr_db, p):
    if p.get('type', 'minmax') == 'standard':
        clip = float(p.get('clip', 3.0))
        z = (arr_db - p['mean']) / (p['std'] + 1e-8)
        z = np.clip(z, -clip, clip)
        return (z + clip) / (2.0 * clip)
    lo, hi = p['min'], p['max']
    return (arr_db - lo) / (hi - lo + 1e-8)


def _align_pred_to_ref_np(pred_norm, src_norm, ref_norm, n_freqs):
    pred_map = _split_raw_target_map(pred_norm, n_freqs)  # 변경: 예측 벡터를 8채널 dict로 분리
    aligned = []
    for key in TARGET_KEYS:
        raw = _norm_to_db_np(pred_map[key], src_norm[key]) if key.endswith('_db') else _norm_to_db_np(pred_map[key], src_norm[key])  # 변경: 공통 helper로 raw 복원
        aligned.append(_db_to_norm_np(raw, ref_norm[key]))  # 변경: ref 기준으로 모든 채널 재정규화
    return np.concatenate(aligned, axis=1).astype(np.float32)


def _batch_surrogate_eval(models, X_bin, device, chunk=512,
                          fold_norm_params=None, ref_norm_params=None, n_freqs=None):
    """앙상블 forward 예측 → (N, output_dim) numpy."""
    models_eval = [m.eval() for m in models]
    preds = []
    with torch.no_grad():
        for s in range(0, len(X_bin), chunk):
            t = torch.FloatTensor(X_bin[s:s+chunk]).unsqueeze(1).to(device)
            per_model = []
            for i, m in enumerate(models):
                p_i = m(t).cpu().numpy()
                if (fold_norm_params is not None and ref_norm_params is not None and
                        n_freqs is not None and i < len(fold_norm_params)):
                    p_i = _align_pred_to_ref_np(
                        p_i, fold_norm_params[i], ref_norm_params, n_freqs
                    )
                per_model.append(p_i)
            preds.append(np.mean(np.stack(per_model, axis=0), axis=0))
    return np.concatenate(preds, axis=0)                  # (N, output_dim)


def _fitness_from_pred(pred_norm, target_norm, n_freqs,
                       pb_mask, sb_mask, fill_arr,
                       bw_penalty_weight=40.0,
                       sb_penalty_weight=25.0,
                       rl_penalty_weight=10.0):
    """
    비교 공정: inverse_test.py asymmetric_physics_loss 동일 계수 사용.
    pred_norm / target_norm: (N, 3*n_freqs) numpy float32
    반환: (N,) fitness float32
    """
    N = pred_norm.shape[0]
    pred_map = _split_raw_target_map(pred_norm, n_freqs)  # 변경: 8채널 예측에서 dB 채널만 이름으로 추출
    target_map = {key: target_norm[i * n_freqs:(i + 1) * n_freqs] for i, key in enumerate(TARGET_KEYS)}  # 변경: target도 8채널 dict로 분리
    p11 = pred_map['s11_db']
    p21 = pred_map['s21_db']
    p22 = pred_map['s22_db']
    t11 = target_map['s11_db']
    t21 = target_map['s21_db']
    t22 = target_map['s22_db']

    loss  = np.mean((p21[:, pb_mask] - t21[pb_mask])**2,            axis=1) * 60.0
    loss += np.mean(np.maximum(0, p21[:, sb_mask] - t21[sb_mask])**2, axis=1) * 15.0
    loss += np.mean((p11[:, pb_mask] - t11[pb_mask])**2,            axis=1) * 20.0
    loss += np.mean((p22[:, pb_mask] - t22[pb_mask])**2,            axis=1) * 20.0
    loss += np.mean((p11[:, sb_mask] - t11[sb_mask])**2,            axis=1) *  3.0
    loss += np.mean((p22[:, sb_mask] - t22[sb_mask])**2,            axis=1) *  3.0

    # 스펙 위반 페널티: 입력 BW/IL/RL 목표를 더 강하게 강제
    pb_s21_violation = np.mean(np.maximum(0, t21[pb_mask] - p21[:, pb_mask])**2, axis=1)
    sb_s21_violation = np.mean(np.maximum(0, p21[:, sb_mask] - t21[sb_mask])**2, axis=1)
    pb_rl11_violation = np.mean(np.maximum(0, p11[:, pb_mask] - t11[pb_mask])**2, axis=1)
    pb_rl22_violation = np.mean(np.maximum(0, p22[:, pb_mask] - t22[pb_mask])**2, axis=1)
    loss += bw_penalty_weight * pb_s21_violation
    loss += sb_penalty_weight * sb_s21_violation
    loss += rl_penalty_weight * (pb_rl11_violation + pb_rl22_violation)

    # fill 페널티 (7~72%)
    dp    = (np.maximum(0.0, fill_arr - 0.72)**2 +
             np.maximum(0.0, 0.07  - fill_arr)**2) * 8000.0
    return (loss + dp).astype(np.float32)


def stage2_bo_de_validation(models, norm_params, freqs, device, output_path,
                             fold_norm_params=None,
                             budget=500, n_init=100, seed=42,
                             target_fc_ghz=7.0, target_bw_pct=15.0,
                             target_il_db=-1.0, target_sb_db=-20.0,
                             target_rl_db=-10.0, transition_scale=1.2,
                             bw_penalty_weight=40.0,
                             sb_penalty_weight=25.0,
                             rl_penalty_weight=10.0):
    """
    Stage-2: 훈련된 앙상블 surrogate 를 활용해
    BO (RF-EI) 와 DE 를 동일 조건(seed=42, budget=500)으로 비교.

    비교 기준: BPSO-like 랜덤 탐색(budget=500) 도 함께 실행.
    결과를 stage2_results.json / stage2_best_layouts.npz 로 저장.
    """
    if not _BO_AVAILABLE:
        print("  ⚠ scikit-learn / scipy 없음 → Stage-2 스킵 (pip install scikit-learn scipy)")
        return

    print(TITLE)
    print("STAGE 2: BO & DE Validation (Budget=500, seed=42)")
    print("  [주의] Stage-2 fitness는 surrogate 예측 기반이며 실제 EM 성능을 직접 보장하지 않습니다.")
    print(TITLE)

    n_freqs  = N_FREQS
    out_dim  = OUTPUT_DIM

    # ── 목표 스펙 생성 (사용자 입력 fc/BW/IL/RL/SB 기준) ───────────────────
    fc, bw_pct = float(target_fc_ghz), float(target_bw_pct)
    half       = fc * bw_pct / 100.0
    fl, fh     = fc - half, fc + half
    pb_mask    = (freqs >= fl) & (freqs <= fh)
    tl, tr     = fl / max(1e-6, float(transition_scale)), fh * float(transition_scale)
    sb_mask    = (freqs <= tl) | (freqs >= tr)

    from scipy.ndimage import gaussian_filter1d
    t11_db = gaussian_filter1d(np.where(pb_mask, target_rl_db, -3.0), 2).astype(np.float32)
    t21_db = gaussian_filter1d(np.where(pb_mask, target_il_db, target_sb_db), 1).astype(np.float32)
    t22_db = t11_db.copy()
    print(f"  [Target] fc={fc:.3f} GHz  BW={bw_pct:.2f}%  "
          f"IL={target_il_db:.1f} dB  RL={target_rl_db:.1f} dB  SB={target_sb_db:.1f} dB")

    # 정규화 (norm_type 자동 감지)
    def _norm_ch(arr, key):
        p = norm_params[key]
        if p.get('type', 'minmax') == 'standard':
            z   = (arr - p['mean']) / (p['std'] + 1e-8)
            z   = np.clip(z, -p['clip'], p['clip'])
            return (z + p['clip']) / (2.0 * p['clip'])
        else:
            lo, hi = p['min'], p['max']
            return (arr - lo) / (hi - lo + 1e-8)

    target_map = {key: np.zeros_like(freqs, dtype=np.float32) for key in TARGET_KEYS}  # 변경: Stage-2 target도 8채널 형식으로 구성
    target_map['s11_db'] = t11_db
    target_map['s12_db'] = t21_db
    target_map['s21_db'] = t21_db
    target_map['s22_db'] = t22_db
    target_norm = np.concatenate([_norm_ch(target_map[key], key) for key in TARGET_KEYS]).astype(np.float32)  # 변경: degree 채널은 0 기본값으로 정규화

    def batch_eval(pop_bin):
        """pop_bin (N,25,25) → fitness (N,)"""
        pred = _batch_surrogate_eval(
            models, pop_bin, device,
            fold_norm_params=fold_norm_params,
            ref_norm_params=norm_params,
            n_freqs=n_freqs
        )
        fill = pop_bin.mean(axis=(1, 2))
        return _fitness_from_pred(pred, target_norm, n_freqs,
                                  pb_mask, sb_mask, fill,
                                  bw_penalty_weight=bw_penalty_weight,
                                  sb_penalty_weight=sb_penalty_weight,
                                  rl_penalty_weight=rl_penalty_weight)

    results_summary = {}

    # ════════════════════════════════════════════════════════════
    # [A] Random Baseline (동일 budget)
    # ════════════════════════════════════════════════════════════
    print(f"\n[A] Random Baseline  budget={budget}")
    np.random.seed(seed)
    rand_pop = _make_random_pop(budget, seed=seed)
    rand_fit = batch_eval(rand_pop)
    bi = int(np.argmin(rand_fit))
    results_summary['Random'] = {
        'best_fitness': float(rand_fit[bi]),
        'best_layout_idx': bi,
    }
    rand_best_layout = rand_pop[bi].copy()
    print(f"  best fitness = {rand_fit[bi]:.6f}")

    # ════════════════════════════════════════════════════════════
    # [B] Differential Evolution (binary DE/rand/1/bin + repair)
    # F=0.8, CR=0.9, POP=100, budget=500
    # ════════════════════════════════════════════════════════════
    print(f"\n[B] Differential Evolution  F=0.8 CR=0.9 POP=100 budget={budget}")
    np.random.seed(seed)
    POP_DE   = 100
    F_SCALE  = 0.8
    CR_DE    = 0.9
    LS_EVERY = 5

    # 초기 집단: 모자이크 랜덤으로 통일 (뭉탱이 warm-start 제거)
    de_bin  = _make_random_pop(POP_DE, seed=seed + 101)
    de_cont = np.clip(de_bin + np.random.normal(0, 0.05, de_bin.shape), 0.0, 1.0).astype(np.float32)

    de_fit     = batch_eval(de_bin)
    de_evals   = POP_DE
    de_best_fi = float(de_fit.min())
    de_best_layout = de_bin[int(np.argmin(de_fit))].copy()
    de_no_imp  = 0
    de_history = [(de_evals, de_best_fi)]
    de_gen     = 0
    de_n_improved = 0        # [추가] 선택 갱신 횟수 (검증용)
    de_init_best  = de_best_fi  # [추가] 초기 best (수렴 확인용)

    while de_evals < budget:
        de_gen += 1
        idx_base = np.arange(POP_DE)
        r_idx    = np.array([
            np.random.choice(idx_base[idx_base != i], 3, replace=False)
            for i in range(POP_DE)
        ])
        mutant = np.clip(
            de_cont[r_idx[:, 0]] + F_SCALE * (de_cont[r_idx[:, 1]] - de_cont[r_idx[:, 2]]),
            0.0, 1.0)
        cross = np.random.rand(POP_DE, GRID_SIZE, GRID_SIZE) < CR_DE
        j_r   = np.random.randint(GRID_SIZE, size=POP_DE)
        j_c   = np.random.randint(GRID_SIZE, size=POP_DE)
        cross[np.arange(POP_DE), j_r, j_c] = True
        trials_cont = np.where(cross, mutant, de_cont)
        trials_bin  = (trials_cont >= 0.5).astype(np.float32)
        _apply_port_constraints(trials_bin)

        n_eval = min(POP_DE, budget - de_evals)
        t_fit  = batch_eval(trials_bin[:n_eval])
        de_evals += n_eval
        # [BUG FIX] 체인 boolean 인덱싱 대신 정수 인덱스로 안전하게 갱신
        imp_mask = t_fit < de_fit[:n_eval]          # (n_eval,) bool
        imp_idx  = np.where(imp_mask)[0]            # 절대 인덱스 ([0, n_eval) 범위)
        if imp_idx.size:
            de_cont[imp_idx] = trials_cont[imp_idx]
            de_bin[imp_idx]  = trials_bin[imp_idx]
            de_fit[imp_idx]  = t_fit[imp_mask]
            de_n_improved   += int(imp_idx.size)

        bi = int(np.argmin(de_fit))
        if de_fit[bi] < de_best_fi:
            de_best_fi     = float(de_fit[bi])
            de_best_layout = de_bin[bi].copy()
            de_no_imp      = 0
        else:
            de_no_imp += 1

        # 로컬서치 (엘리트 상위 2개, budget 여유 있을 때)
        if de_gen % LS_EVERY == 0 and de_evals + 4 <= budget:
            for ei in np.argsort(de_fit)[:2]:
                cands = np.tile(de_bin[ei][np.newaxis], (4, 1, 1))
                for k in range(4):
                    nr = np.random.randint(1, 6)
                    rs = np.random.randint(0, GRID_SIZE, nr)
                    cs = np.random.randint(0, GRID_SIZE, nr)
                    cands[k, rs, cs] = 1.0 - cands[k, rs, cs]
                _apply_port_constraints(cands)
                cf = batch_eval(cands)
                de_evals += 4
                bk = int(np.argmin(cf))
                if cf[bk] < de_fit[ei]:
                    de_bin[ei]  = cands[bk]
                    de_cont[ei] = np.clip(cands[bk] + np.random.normal(0, 0.05, cands[bk].shape), 0.0, 1.0)
                    de_fit[ei]  = cf[bk]
                    if cf[bk] < de_best_fi:
                        de_best_fi     = float(cf[bk])
                        de_best_layout = cands[bk].copy()
                        de_no_imp      = 0

        # 다양성 재초기화
        if de_no_imp >= 50 and de_evals + POP_DE // 4 <= budget:
            nr   = POP_DE // 4
            nb   = _make_random_pop(nr)
            wi   = np.argsort(de_fit)[-nr:]
            de_bin[wi]  = nb; de_cont[wi] = np.clip(nb + np.random.normal(0, 0.05, nb.shape), 0.0, 1.0)
            nf = batch_eval(nb); de_evals += nr
            de_fit[wi]  = nf; de_no_imp  = 0

        de_history.append((de_evals, de_best_fi))
        if de_gen % 10 == 0:
            print(f"  DE Gen {de_gen:4d}  evals={de_evals:4d}/{budget}  "
                  f"best(raw)={de_best_fi:.6f}  개선횟수={de_n_improved}")

    results_summary['DE'] = {
        'best_fitness': de_best_fi,
        'history':      de_history,
    }
    print(f"  DE 완료  best_fitness(raw)={de_best_fi:.6f}  "
          f"초기→최종: {de_init_best:.6f}→{de_best_fi:.6f}  총갱신={de_n_improved}")

    # ════════════════════════════════════════════════════════════
    # [C] Bayesian Optimization (RF-EI, batch=20, n_init=100)
    # ════════════════════════════════════════════════════════════
    print(f"\n[C] Bayesian Optimization  n_init={n_init} budget={budget}")
    np.random.seed(seed)
    N_CAND_BO = 5000
    N_TREES   = 60
    BATCH_EI  = 20

    bo_pop_init = _make_random_pop(n_init, seed=seed)
    bo_fit_init = batch_eval(bo_pop_init)
    bo_evals    = n_init
    bo_bi       = int(np.argmin(bo_fit_init))
    bo_best_fi  = float(bo_fit_init[bo_bi])
    bo_best_layout = bo_pop_init[bo_bi].copy()

    X_obs = bo_pop_init.reshape(n_init, -1).astype(np.float32)
    y_obs = bo_fit_init.astype(np.float32)
    bo_history  = [(bo_evals, bo_best_fi)]
    bo_iter     = 0

    while bo_evals < budget:
        bo_iter += 1

        rf = _RF(n_estimators=N_TREES, min_samples_leaf=2,
                 max_features='sqrt', n_jobs=-1, random_state=seed)
        rf.fit(X_obs, y_obs)

        # 후보 생성: 모자이크 랜덤만 사용 (chunky warm-start 제거)
        cand_pop  = _make_random_pop(N_CAND_BO)
        cand_flat = cand_pop.reshape(N_CAND_BO, -1).astype(np.float32)

        # RF tree별 예측 → μ / σ
        tree_preds = np.array([t.predict(cand_flat) for t in rf.estimators_],
                               dtype=np.float32)           # (n_trees, N_CAND_BO)
        mu    = tree_preds.mean(axis=0)
        sigma = tree_preds.std(axis=0) + 1e-9
        y_star = float(y_obs.min())
        Z      = (y_star - mu) / sigma
        ei     = sigma * (_sp_norm.pdf(Z) + Z * _sp_norm.cdf(Z))

        # 다양성 보장: top 후보 중 해밍 거리 greedy 선택
        top_idx  = np.argsort(-ei)[:BATCH_EI * 4]
        top_flat = cand_flat[top_idx]
        # 최근 20개 관측 대비 최소 해밍 거리 최대화
        ref = X_obs[-min(20, len(X_obs)):]
        dists = np.sum(top_flat[:, None, :] != ref[None, :, :], axis=2).min(axis=1)
        div_idx  = np.argsort(-dists)[:BATCH_EI]
        sel_pidx = top_idx[div_idx]

        n_ev  = min(len(sel_pidx), budget - bo_evals)
        sel   = cand_pop[sel_pidx[:n_ev]]
        s_fit = batch_eval(sel)
        bo_evals += n_ev

        X_obs = np.concatenate([X_obs, sel.reshape(n_ev, -1).astype(np.float32)], axis=0)
        y_obs = np.concatenate([y_obs, s_fit[:n_ev]], axis=0)

        for k in range(n_ev):
            if s_fit[k] < bo_best_fi:
                bo_best_fi     = float(s_fit[k])
                bo_best_layout = sel[k].copy()

        bo_history.append((bo_evals, bo_best_fi))
        if bo_iter % 5 == 0:
            print(f"  BO Iter {bo_iter:4d}  evals={bo_evals:4d}/{budget}  best={bo_best_fi:.6f}")

    results_summary['BO'] = {
        'best_fitness': bo_best_fi,
        'history':      bo_history,
    }
    print(f"  BO 완료  best_fitness={bo_best_fi:.6f}")

    # ════════════════════════════════════════════════════════════
    # 결과 요약 출력 & 저장
    # ════════════════════════════════════════════════════════════
    print(TITLE)
    print("STAGE 2 RESULTS (budget=500, seed=42)")
    print(f"{'Algorithm':>10}  {'Best Fitness (raw)':>18}  {'Note':}")
    print(SEP)
    for alg in ['Random', 'DE', 'BO']:
        fi = results_summary[alg]['best_fitness']
        print(f"  {alg:>8}  {fi:>18.6f}  (×100 → {fi*100:.4f})")
    print(TITLE)

    # JSON 저장
    json_safe = copy.deepcopy(results_summary)
    for k in json_safe:
        if 'history' in json_safe[k]:
            json_safe[k]['history'] = [[int(e), float(v)]
                                       for e, v in json_safe[k]['history']]
    with open(output_path / 'stage2_results.json', 'w', encoding='utf-8') as f:
        json.dump(json_safe, f, indent=2)

    # 베스트 레이아웃 저장
    np.savez_compressed(
        output_path / 'stage2_best_layouts.npz',
        random=rand_best_layout,
        de=de_best_layout,
        bo=bo_best_layout,
        target_norm=target_norm,
        pb_mask=pb_mask.astype(np.uint8),
        sb_mask=sb_mask.astype(np.uint8),
    )
    print(f"  저장: stage2_results.json  stage2_best_layouts.npz  →  {output_path}")
    return results_summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── 데이터 경로 ──────────────────────────────────────────────────────────
    parser.add_argument('--layouts',    type=str,
                        default='/scratch/home/bhj101500/25x25_data/layouts.npz')
    parser.add_argument('--sparams',    type=str,
                        default='/scratch/home/bhj101500/25x25_data/s_params.npz')
    parser.add_argument('--output-dir', type=str, default=None)

    # ── 학습 하이퍼파라미터 ──────────────────────────────────────────────────
    parser.add_argument('--stage1-epochs', type=int,   default=200)
    parser.add_argument('--batch-size',    type=int,   default=None,
                        help='수동 배치 크기 (기본값 None = VRAM 자동 감지 강제)')
    parser.add_argument('--auto-batch',    action='store_true', default=True,
                        help='기본 동작이므로 무시됨 (항상 VRAM 기반 자동 설정 동작)')
    parser.add_argument('--kfolds',        type=int,   default=5)
    parser.add_argument('--lr',            type=float, default=3e-4)
    parser.add_argument('--patience',      type=int,   default=30)
    parser.add_argument('--num-workers',   type=int,   default=4,
                        help='DataLoader 워커 수 (Windows=0 자동 고정)')

    # ── 아키텍처 옵션 ────────────────────────────────────────────────────────
    parser.add_argument('--stem-type',      type=str, default='small',
                        choices=['standard', 'small'],
                        help="'standard': 7×7 stride-1 | "
                             "'small': 3×3×2 최적 stem (기본값, 25×25 권장)")
    parser.add_argument('--pretrained',     action='store_true', default=False,
                        help='ImageNet 사전학습 가중치 사용 (기본: 미사용, scratch 학습)')
    parser.add_argument('--pretrained-path', type=str, default=None,
                        help='로컬 사전학습 가중치 경로 (.pt / .pth). '
                             '오프라인 환경 또는 도메인 파인튜닝용.')

    # ── 정규화 옵션 ──────────────────────────────────────────────────────────
    parser.add_argument('--norm-type', type=str, default='standard',
                        choices=['minmax', 'standard'],
                        help="'minmax': Min-Max [0,1] | "
                             "'standard': Z-score→clip(±3σ)→[0,1] (기본값) "
                             "(S21 스케일 편차 클 때 권장)")

    # ── Stage-2 옵션 ─────────────────────────────────────────────────────────
    parser.add_argument('--skip-stage2',  action='store_true',
                        help='Stage-2 BO/DE 검증 스킵')
    parser.add_argument('--bo-de-budget', type=int, default=500)
    parser.add_argument('--target-fc-ghz', type=float, default=7.0,
                        help='Stage-2 목표 중심 주파수 (GHz)')
    parser.add_argument('--target-bw-pct', type=float, default=15.0,
                        help='Stage-2 목표 대역폭 (fc 대비 %)')
    parser.add_argument('--target-il-db', type=float, default=-1.0,
                        help='Stage-2 통과대역 S21 목표(dB)')
    parser.add_argument('--target-sb-db', type=float, default=-20.0,
                        help='Stage-2 저지대역 S21 목표(dB)')
    parser.add_argument('--target-rl-db', type=float, default=-10.0,
                        help='Stage-2 통과대역 S11/S22 목표(dB)')
    parser.add_argument('--transition-scale', type=float, default=1.2,
                        help='저지대역 경계 비율(>1.0 권장)')
    parser.add_argument('--bw-penalty-weight', type=float, default=40.0,
                        help='통과대역(BW) 위반 페널티 가중치')
    parser.add_argument('--sb-penalty-weight', type=float, default=25.0,
                        help='저지대역 위반 페널티 가중치')
    parser.add_argument('--rl-penalty-weight', type=float, default=10.0,
                        help='통과대역 반사손실 위반 페널티 가중치')

    args = parser.parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    print(TITLE)
    print("FORWARD ENSEMBLE TRAINING — DenseNet121 × K-Fold  (v3)")
    print(f"  그리드: {GRID_SIZE}×{GRID_SIZE}  |  주파수: {N_FREQS} pts  "
          f"|  output_dim: {OUTPUT_DIM}")
    print(TITLE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        prop = torch.cuda.get_device_properties(0)
        print(f"  GPU  : {prop.name}")
        print(f"  VRAM : {prop.total_memory / 1e9:.1f} GB")

    # ── Batch size 결정 ──────────────────────────────────────────────────────
    batch_size = auto_batch_size(device, manual_batch=args.batch_size)

    # ── 출력 경로 ────────────────────────────────────────────────────────────
    script_dir = Path(__file__).parent.resolve()
    if args.output_dir:
        output_path = Path(args.output_dir)
    elif os.name == 'nt':
        output_path = script_dir.parent.parent / 'model' / 'test_model'
    else:
        output_path = script_dir / 'test_model'
    os.makedirs(output_path, exist_ok=True)
    print(f"출력 경로: {output_path}")

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    print(SEP)
    print("DATA LOADING")
    X, y_raw, freqs, norm_params, target_keys = load_data(
        Path(args.layouts), Path(args.sparams),
        norm_type=args.norm_type)

    # ── Stage 1 학습 ─────────────────────────────────────────────────────────
    t_start = time.time()
    output_dim, fold_results = stage1_forward_ensemble(
        X, y_raw, freqs, device, output_path,
        max_epochs    = args.stage1_epochs,
        batch_size    = batch_size,
        lr            = args.lr,
        patience      = args.patience,
        kfolds        = args.kfolds,
        num_workers   = args.num_workers,
        norm_type     = args.norm_type,
        stem_type     = args.stem_type,
        use_pretrained= args.pretrained,
        pretrained_path = args.pretrained_path,
    )
    total_time = time.time() - t_start

    # ── model_config.json 저장 (inverse_test.py 호환) ────────────────────────
    config = {
        'schema_version':     1,
        'model_type':         'forward_surrogate',
        'backbone':           'densenet121',
        'stem_type':          args.stem_type,          # ★ 신규
        'grid_size':          GRID_SIZE,
        'input_size':         [GRID_SIZE, GRID_SIZE],
        'target_keys':        target_keys,
        'output_format':      'db_only',
        'output_dim':         output_dim,
        'freq_points':        len(freqs),
        'freq_range_ghz':     [float(freqs[0]), float(freqs[-1])],
        'frequencies_file':   'frequencies.npy',
        'checkpoint_format':  f'ensemble/{CHECKPOINT_TEMPLATE}',
        'config_name':        COMMON_CONFIG_NAME,
        'normalization':      norm_params,             # minmax or standard
        'fold_normalizations': [r['norm_params'] for r in fold_results],
        'normalization_type': args.norm_type,          # ★ 신규
        'k_folds':            args.kfolds,
        'batch_size':         batch_size,
        'max_epochs':         args.stage1_epochs,
        'training_time_min':  round(total_time / 60, 1),
        'fold_results':       fold_results,
        'spec': {
            'board_size_mm':      [20.0, 20.0],
            'boundary_offset_mm': 5.0,
            'pixel_size_mm':      0.4,
            'port_width_mm':      1.2,
            'port_length_mm':     3.0,
            'port_y_mm':          10.0,
            'substrate_h_mm':     1.2,
            'er':                 4.0,
            'tand':               0.013,
        },
    }
    for name in [COMMON_CONFIG_NAME, LEGACY_CONFIG_NAME]:
        with open(output_path / name, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    np.save(output_path / 'frequencies.npy', freqs)

    print(TITLE)
    print("STAGE 1 TRAINING COMPLETE!")
    print(f"  시간  : {total_time/60:.1f} min")
    print(f"  출력  : {output_path}")
    print(f"  stem  : {args.stem_type}  |  norm: {args.norm_type}  "
          f"|  batch: {batch_size}")
    print(f"  저장  : ensemble/{CHECKPOINT_TEMPLATE}")
    print(f"          {COMMON_CONFIG_NAME} + {LEGACY_CONFIG_NAME}")
    print(f"          ensemble_meta.json + frequencies.npy")
    print(TITLE)

    # ── Stage 2: BO / DE 검증 ────────────────────────────────────────────────
    if not args.skip_stage2:
        print(SEP)
        print("STAGE 2: 앙상블 모델 재로드 중...")
        ensemble_dir  = output_path / 'ensemble'
        # glob 금지: 이름·개수 불일치 시 조용히 2개만 로드되는 사고 방지.
        # forward_fold0..{k-1} 를 fold 인덱스 순으로 명시 로드 + 정규화 메타와 1:1 대응.
        fold_files = _expected_fold_checkpoints(ensemble_dir, args.kfolds)
        missing = [ck for ck in fold_files
                   if not ck.is_file() or ck.stat().st_size < 64]
        if missing:
            print(f"  ❌ Stage-2에 필요한 체크포인트 {len(missing)}개 누락/손상:")
            for ck in missing:
                ex, sz = ck.exists(), (ck.stat().st_size if ck.exists() else 0)
                print(f"     {ck}  exists={ex}  size={sz}")
            found = sorted(ensemble_dir.glob('*.pt'))
            print(f"  ensemble/*.pt 실제 ({len(found)}개):")
            for p in found:
                print(f"     {p.name}  {p.stat().st_size}")
            raise RuntimeError(
                "Stage-1이 저장한 forward_fold0..N.pt 가 ensemble/에 없습니다. "
                "출력 경로·다른 작업 덮어쓰기·스크립트 복사본 불일치를 확인하세요.")
        loaded_models = []
        for ff in fold_files:
            m = DenseNet25x25(output_dim=output_dim,
                              use_pretrained=False,
                              stem_type=args.stem_type)
            try:
                ck = torch.load(ff, map_location=device, weights_only=True)
            except Exception:
                ck = torch.load(ff, map_location=device, weights_only=False)
            m.load_state_dict(ck)
            m.to(device).eval()
            for p in m.parameters():
                p.requires_grad = False
            loaded_models.append(m)
        print(f"  {len(loaded_models)}개 fold 모델 로드 완료")
        fold_norm_params = [r['norm_params'] for r in sorted(fold_results, key=lambda d: d['fold'])]

        stage2_bo_de_validation(
            loaded_models, norm_params, freqs, device, output_path,
            fold_norm_params=fold_norm_params,
            budget=args.bo_de_budget,
            target_fc_ghz=args.target_fc_ghz,
            target_bw_pct=args.target_bw_pct,
            target_il_db=args.target_il_db,
            target_sb_db=args.target_sb_db,
            target_rl_db=args.target_rl_db,
            transition_scale=args.transition_scale,
            bw_penalty_weight=args.bw_penalty_weight,
            sb_penalty_weight=args.sb_penalty_weight,
            rl_penalty_weight=args.rl_penalty_weight,
        )
    else:
        print("  Stage-2 스킵 (--skip-stage2)")


if __name__ == '__main__':
    main()
