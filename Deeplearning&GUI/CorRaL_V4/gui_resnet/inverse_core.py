#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""현재 GUI를 그대로 재사용하면서 ResNet 원본 알고리즘을 물리는 전용 코어."""  # 변경: 별도 테스트용 코어 추가

from __future__ import annotations
import traceback
import torch


import argparse  # 변경: ResNet 원본 알고리즘 인자 네임스페이스 구성용
import sys  # 변경: 외부 폴더 모듈 경로 등록용
from pathlib import Path  # 변경: 외부 파일 경로 해석용

import numpy as np  # 변경: surrogate adapter / target 해석용
import importlib.util


_THIS_DIR = Path(__file__).resolve().parent  # 변경: 새 테스트 폴더 기준 경로
_BASE_GUI_DIR = _THIS_DIR / "_base_gui"  # 변경: CorRaL 내부에 복사한 하이브리드 GUI 코어 참조
_RESNET_DIR = _THIS_DIR / "_resnet_reference"  # 변경: CorRaL 내부에 복사한 ResNet 원본 알고리즘 참조

for _extra_path in (_BASE_GUI_DIR, _RESNET_DIR):  # 변경: 동적 import가 참조하는 로컬 모듈 경로 추가
    _extra_str = str(_extra_path)
    if _extra_str not in sys.path:
        sys.path.insert(0, _extra_str)


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))  # 변경: 주석에 묻힐 수 있는 spec 생성을 명시적으로 복원
    if spec is None or spec.loader is None:  # 변경: 잘못된 경로면 즉시 중단
        raise ImportError(f"Cannot load module: {file_path}")
    module = importlib.util.module_from_spec(spec)  # 변경: 로드 대상 모듈 생성
    sys.modules[module_name] = module  # 변경: dataclass/annotation 해석을 위해 sys.modules 선등록
    spec.loader.exec_module(module)  # 변경: 실제 모듈 실행을 명시적으로 복원
    return module  # 변경: 아래 깨진 라인에 의존하지 않고 바로 반환


_base_core = _load_module("_inverse_gui_base_core", _BASE_GUI_DIR / "inverse_core.py")  # 변경: 현재 GUI가 기대하는 공통 유틸/로더 재사용
_resnet_algo = _load_module("_inverse_resnet_algo", _RESNET_DIR / "inverse_design_25x25.py")  # 변경: 네 원래 ResNet 알고리즘 재사용


for _name, _value in _base_core.__dict__.items():  # 변경: 현재 GUI가 쓰는 이름들을 그대로 재수출
    if _name.startswith("__"):
        continue
    globals()[_name] = _value



_ACTIVE_SPLIT_LINE_CONFIG = None  # 변경: retrieval seed 평가 때 현재 GUI 분리선 설정을 공유


class _GUIEnsembleSurrogate:
    """현재 GUI가 로드한 forward ensemble을 네 원래 알고리즘 인터페이스에 맞춘 어댑터."""  # 변경: current GUI <-> old ResNet algorithm 브리지

    def __init__(self, forward_models, config, device, frequencies, chunk_size=512):
        self.models = forward_models  # 변경: 기존 GUI가 로드한 모델 그대로 사용
        self.config = config or {}
        self.device = device
        self.frequencies = np.asarray(frequencies, dtype=np.float32)
        self.chunk_size = max(1, int(chunk_size))
        self.target_keys = self.config.get("target_keys", ["s11_db", "s21_db", "s22_db"])
        self.normalization = self.config.get("normalization", {})
        self.normalization_type = self.config.get("normalization_type", "minmax")

    def _norm_mode(self, key: str) -> str:
        params = self.normalization.get(key, {})
        return params.get("type", self.normalization_type)  # 변경: current GUI의 nested/legacy 정규화 형식 모두 수용

    def denormalize_predictions(self, pred_norm: np.ndarray) -> np.ndarray:
        pred_norm = np.asarray(pred_norm, dtype=np.float32)
        n_freq = len(pred_norm) // 3
        outputs = []
        for idx, key in enumerate(["s11_db", "s21_db", "s22_db"]):
            chunk = pred_norm[idx * n_freq:(idx + 1) * n_freq]
            params = self.normalization.get(key, {})
            mode = self._norm_mode(key)
            if mode == "standard":
                clip = float(params.get("clip", 3.0))
                z = np.clip(chunk, 0.0, 1.0) * (2.0 * clip) - clip
                outputs.append(z * float(params["std"]) + float(params["mean"]))  # 변경: standard 정규화 역변환 지원
            elif "s11_db" in self.normalization:
                lo = float(params.get("min", -140.0))
                hi = float(params.get("max", 0.0))
                outputs.append(np.clip(chunk, 0.0, 1.0) * (hi - lo) + lo)  # 변경: nested minmax 역변환
            elif "s11_min" in self.normalization:
                lo = float(self.normalization[key.replace("_db", "_min")])
                hi = float(self.normalization[key.replace("_db", "_max")])
                outputs.append(np.clip(chunk, 0.0, 1.0) * (hi - lo) + lo)  # 변경: legacy flat minmax 역변환
            else:
                lo = float(self.normalization.get("s_min", -140.0))
                hi = float(self.normalization.get("s_max", 0.0))
                outputs.append(np.clip(chunk, 0.0, 1.0) * (hi - lo) + lo)  # 변경: global minmax 역변환
        return np.concatenate(outputs).astype(np.float32)

    def normalize_targets(self, target_db: np.ndarray) -> np.ndarray:
        target_db = np.asarray(target_db, dtype=np.float32)
        n_freq = len(target_db) // 3
        outputs = []
        for idx, key in enumerate(["s11_db", "s21_db", "s22_db"]):
            chunk = target_db[idx * n_freq:(idx + 1) * n_freq]
            params = self.normalization.get(key, {})
            mode = self._norm_mode(key)
            if mode == "standard":
                clip = float(params.get("clip", 3.0))
                z = (chunk - float(params["mean"])) / (float(params["std"]) + 1e-8)
                z = np.clip(z, -clip, clip)
                outputs.append((z + clip) / (2.0 * clip))  # 변경: old GD가 쓰는 normalize_targets 제공
            elif "s11_db" in self.normalization:
                lo = float(params.get("min", -140.0))
                hi = float(params.get("max", 0.0))
                outputs.append((chunk - lo) / (hi - lo + 1e-8))  # 변경: nested minmax 정방향 변환
            elif "s11_min" in self.normalization:
                lo = float(self.normalization[key.replace("_db", "_min")])
                hi = float(self.normalization[key.replace("_db", "_max")])
                outputs.append((chunk - lo) / (hi - lo + 1e-8))  # 변경: legacy flat minmax 정방향 변환
            else:
                lo = float(self.normalization.get("s_min", -140.0))
                hi = float(self.normalization.get("s_max", 0.0))
                outputs.append((chunk - lo) / (hi - lo + 1e-8))  # 변경: global minmax 정방향 변환
        return np.concatenate(outputs).astype(np.float32)

    def predict(self, layouts: np.ndarray, return_std: bool = True):
        layouts = np.asarray(layouts, dtype=np.float32)
        if layouts.ndim == 2:
            layouts = layouts[None, ...]
        stacked = []
        with torch.no_grad():
            for model in self.models:
                chunks = []
                for start in range(0, len(layouts), self.chunk_size):
                    end = min(start + self.chunk_size, len(layouts))
                    batch = torch.from_numpy(layouts[start:end]).unsqueeze(1).to(self.device)
                    chunks.append(model(batch).detach().cpu().numpy())
                stacked.append(np.concatenate(chunks, axis=0))
        stacked = np.stack(stacked, axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0) if return_std else np.zeros_like(mean)
        return mean.astype(np.float32), std.astype(np.float32)


def _build_args_namespace(pop_size: int, chunk_size: int):
    """네 원래 ResNet GUI/CLI 기본값을 그대로 담은 args namespace."""  # 변경: old algorithm 하이퍼파라미터 재사용
    args = argparse.Namespace()
    args.chunk_size = max(1, int(chunk_size))
    args.ga_population = int(pop_size)
    args.ga_generations = 120
    args.bpso_particles = int(pop_size)
    args.bpso_iterations = 120
    args.gd_steps = 250
    args.gd_lr = 0.05
    args.dbs_passes = 3
    args.retrieval_topk = int(getattr(_resnet_algo, "RETRIEVAL_TOPK", 16))
    args.retrieval_seeds = int(getattr(_resnet_algo, "RETRIEVAL_SEED_COUNT", 8))
    args.w_s21_passband = float(getattr(_resnet_algo, "DEFAULT_W_S21_PASSBAND", 30.0))
    args.w_s11_passband = float(getattr(_resnet_algo, "DEFAULT_W_S11_PASSBAND", 5.0))
    args.w_s22_passband = float(getattr(_resnet_algo, "DEFAULT_W_S22_PASSBAND", 5.0))
    args.w_s21_stopband = float(getattr(_resnet_algo, "DEFAULT_W_S21_STOPBAND", 10.0))
    return args


class InverseDesignWorker(QThread):
    """현재 GUI 시그니처를 유지하면서 내부 최적화만 네 ResNet 알고리즘으로 교체한 워커."""  # 변경: current GUI 호환 래퍼 워커

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    conv_update = pyqtSignal(str, int, float)
    layout_update = pyqtSignal(str, object, object)

    _ALGO_MAP = {  # 변경: current GUI uppercase 명칭을 old algorithm lowercase 명칭으로 연결
        "GD": "gd",
        "GA": "ga",
        "BPSO": "bpso",
        "GA_GD": "ga_gd",
        "BPSO_GD": "bpso_gd",
        "GA_DBS": "ga_dbs",
        "BPSO_DBS": "bpso_dbs",
    }

    def __init__(self, forward_models, targets_norm, config, algorithm,
                 device, frequencies, pb_mask, sb_mask, initial_layout=None,
                 pop_size=4096, chunk_size=512, filter_type='lpf',
                 p1_z0=50.0+0j, p2_z0=50.0+0j):
        super().__init__()
        self.forward_models = forward_models  # 변경: 현재 GUI가 로드한 모델 그대로 사용
        self.targets_norm = targets_norm
        self.config = dict(config)  # 변경: worker별 설정을 복사해 포트 임피던스까지 안전하게 보존
        self.algorithm = algorithm
        self.device = device
        self.frequencies = np.asarray(frequencies, dtype=np.float32)
        self.pb_mask = np.asarray(pb_mask, dtype=bool)
        self.sb_mask = np.asarray(sb_mask, dtype=bool)
        self.initial_layout = initial_layout
        self.pop_size = int(pop_size)
        self.chunk_size = int(chunk_size)
        self.filter_type = filter_type
        self.p1_z0 = p1_z0
        self.p2_z0 = p2_z0
        self.config["p1_z0"] = p1_z0  # 변경: fitness가 실제 P1 임피던스를 직접 참조할 수 있게 저장
        self.config["p2_z0"] = p2_z0  # 변경: fitness가 실제 P2 임피던스를 직접 참조할 수 있게 저장
        self.running = True
        self._delegate = None  # 변경: BO/DE는 base core worker로 위임하기 위한 저장소


    def stop(self):
        self.running = False
        if self._delegate is not None:
            self._delegate.stop()  # 변경: 위임된 BO/DE 워커도 중단 전파

    def _callback(self, _algo_name, step, score, hard_layout, soft_layout):
        if not self.running:
            raise RuntimeError("Optimization stopped by user.")  # 변경: old algorithm callback 중단 지원
        # [FIX] Multi-stage algorithm step offset logic for continuous convergence plotting
        offset = 0
        ga_gen = 120
        gd_steps = 250
        dbs_passes = 3
        
        if self.algorithm in {"GA_GD", "GA_DBS", "BPSO_GD", "BPSO_DBS"}:
            if _algo_name in {"gd", "dbs"}:
                offset = ga_gen
            total_total = ga_gen + (gd_steps if "GD" in self.algorithm else dbs_passes)
        elif self.algorithm == "GD":
            total_total = gd_steps
        else:
            total_total = ga_gen
            
        display_step = int(step + offset)
        self.conv_update.emit(self.algorithm, display_step, float(score))
        
        hard_layout = _enforce_split_mask(hard_layout, self.config)
        soft_layout = _enforce_split_mask(soft_layout, self.config) if soft_layout is not None else None
        self.layout_update.emit(self.algorithm, hard_layout, soft_layout)
        
        pct = min(99, int(display_step / max(1, total_total) * 100))
        self.progress.emit(pct, f"{self.algorithm} step={display_step} score={float(score):.4f}")

    def _run_delegate(self):
        """BO/DE는 현재 base core 구현을 그대로 사용."""  # 변경: old ResNet 원본에 없는 알고리즘은 기존 코어 위임
        self._delegate = _base_core.InverseDesignWorker(
            self.forward_models, self.targets_norm, self.config, self.algorithm,
            self.device, self.frequencies, self.pb_mask, self.sb_mask, self.initial_layout,
            self.pop_size, self.chunk_size, self.filter_type,
            p1_z0=self.p1_z0, p2_z0=self.p2_z0,
        )
        self._delegate.progress.connect(lambda *a: self.progress.emit(*a))
        self._delegate.finished.connect(lambda *a: self.finished.emit(*a))
        self._delegate.error.connect(lambda *a: self.error.emit(*a))
        self._delegate.conv_update.connect(lambda *a: self.conv_update.emit(*a))
        self._delegate.layout_update.connect(lambda *a: self.layout_update.emit(*a))
        self._delegate.run()

    def run(self):
        try:
            if self.algorithm in {"BO", "DE"}:
                self._run_delegate()
                return

            algo_name = self._ALGO_MAP.get(self.algorithm)
            if algo_name is None:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")

            self.config = _stabilize_split_line_config(self.config)  # 변경: 포트를 실제로 끊는 분리선이 뽑힐 때까지 재샘플링한 설정으로 고정
            if self.config is not None:
                self.config["split_line_validated"] = True  # 변경: 현재 실행 분리선이 유효성 검사를 통과했음을 기록

            surrogate = _GUIEnsembleSurrogate(  # 변경: current GUI ensemble을 old algorithm surrogate 인터페이스로 래핑
                self.forward_models, self.config, self.device, self.frequencies, self.chunk_size
            )
            surrogate.chunk_size = max(1, self.chunk_size)

            target_db = np.asarray(self.config.get("targets_params", []), dtype=np.float32)
            if target_db.size == 0:
                raise ValueError("targets_params is empty; run_inverse target generation failed.")

            spec = _build_target_spec(self.frequencies, self.config)  # 변경: current GUI spec을 old algorithm spec으로 변환
            args = _build_args_namespace(self.pop_size, self.chunk_size)

            args = _build_args_namespace(self.pop_size, self.chunk_size)  # 변경: old ResNet argument namespace 생성
            global _ACTIVE_SPLIT_LINE_CONFIG
            _ACTIVE_SPLIT_LINE_CONFIG = self.config  # 변경: retrieval seed 평가에도 현재 분리선 설정 공유

            seeds = []
            try:
                layouts_path, sparams_path = _base_core.default_retrieval_paths()
                retrieval_layouts, retrieval_targets, retrieval_freqs = _base_core.load_retrieval_dataset(layouts_path, sparams_path)
                if len(retrieval_freqs) == len(self.frequencies) and np.allclose(retrieval_freqs, self.frequencies):
                    seeds = _resnet_algo.retrieval_seed(
                        target_db, retrieval_layouts, retrieval_targets, self.frequencies, spec,
                        topk=args.retrieval_topk, count=args.retrieval_seeds,
                    )  # 변경: retrieval start도 old ResNet 기준 그대로 사용
                    seeds = [_enforce_split_mask(seed, self.config) for seed in seeds]  # 변경: 실제 시작 seed도 분리선 금지영역이 비워진 상태로 통일
            except Exception:
                seeds = []

            layout, score = _resnet_algo.run_algorithm(  # 변경: 핵심 최적화는 네 원래 ResNet 알고리즘 사용
                algo_name, surrogate, target_db, spec, seeds, args, callback=self._callback
            )
            layout = _enforce_split_mask(layout, self.config)  # 변경: 최종 결과도 분리선 금지영역이 비워진 형상으로 고정
            result = {
                "algorithm": self.algorithm,
                "layout": np.asarray(layout, dtype=np.float32),
                "error": float(score),
                "score": float(score),
            }
            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())


def _resnet_db_target_keys(config: dict | None) -> list[str]:
    """ResNet old algorithm이 쓰는 3채널 dB view 순서."""  # 변경: 8채널 모델에서도 old ResNet core는 3채널 dB만 사용
    target_keys = get_target_keys(config or {})
    return [key for key in ["s11_db", "s21_db", "s22_db"] if key in target_keys]


def _extract_db_view(values: np.ndarray, config: dict | None, n_freq: int) -> np.ndarray:
    """target_keys 기반 벡터에서 old ResNet용 3채널 dB view만 추출."""  # 변경: 8채널 -> 3채널 브리지
    target_keys = get_target_keys(config or {})
    channels = split_prediction_channels(np.asarray(values, dtype=np.float32), target_keys, n_freq)
    if {"s11_real", "s11_imag", "s21_real", "s21_imag", "s22_real", "s22_imag"}.issubset(channels.keys()):
        ordered = []
        for port in ("s11", "s21", "s22"):
            comp = channels[f"{port}_real"] + 1j * channels[f"{port}_imag"]
            ordered.append(20.0 * np.log10(np.maximum(np.abs(comp), 1e-15)))
        return np.concatenate(ordered, axis=-1).astype(np.float32)
    ordered = []
    for key in ["s11_db", "s21_db", "s22_db"]:
        if key not in channels:
            raise KeyError(f"Missing required channel for ResNet algorithm: {key}")
        ordered.append(channels[key])
    return np.concatenate(ordered, axis=-1).astype(np.float32)


def _surrogate_denormalize_predictions(self, pred_norm: np.ndarray) -> np.ndarray:
    pred_norm = np.asarray(pred_norm, dtype=np.float32)
    n_freq = len(self.frequencies)
    if pred_norm.shape[-1] == 3 * n_freq:
        return pred_norm.astype(np.float32)
    target_keys = get_target_keys(self.config)  # 변경: 실제 모델 target_keys 사용
    outputs = []
    for key in target_keys:
        sl = channel_slice(target_keys, key, n_freq)
        chunk = pred_norm[sl]
        params = self.normalization.get(key, {})
        mode = self._norm_mode(key)
        if mode == "standard_direct":
            raw = chunk * float(params["std"]) + float(params["mean"])
            outputs.append(raw)
        elif mode == "standard":
            clip = float(params.get("clip", 3.0))
            z = np.clip(chunk, 0.0, 1.0) * (2.0 * clip) - clip
            raw = z * float(params["std"]) + float(params["mean"])
            outputs.append(np.clip(raw, float(params.get("raw_min", -200.0)), 0.0) if key.endswith("_db") else raw)  # 변경: degree 채널 clip 제외
        elif "s11_db" in self.normalization:
            lo = float(params.get("min", -140.0))
            hi = float(params.get("max", 0.0))
            raw = np.clip(chunk, 0.0, 1.0) * (hi - lo) + lo
            outputs.append(np.clip(raw, lo, 0.0) if key.endswith("_db") else raw)
        elif "s11_min" in self.normalization:
            if key.endswith("_db"):
                lo = float(self.normalization[key.replace("_db", "_min")])
                hi = float(self.normalization[key.replace("_db", "_max")])
            else:
                lo = float(self.normalization[key.replace("_deg", "_min")])
                hi = float(self.normalization[key.replace("_deg", "_max")])
            raw = np.clip(chunk, 0.0, 1.0) * (hi - lo) + lo
            outputs.append(np.clip(raw, lo, 0.0) if key.endswith("_db") else raw)
        else:
            lo = float(self.normalization.get("s_min", -140.0))
            hi = float(self.normalization.get("s_max", 0.0))
            raw = np.clip(chunk, 0.0, 1.0) * (hi - lo) + lo
            outputs.append(np.clip(raw, lo, 0.0) if key.endswith("_db") else raw)
    full = np.concatenate(outputs).astype(np.float32)
    return _extract_db_view(full, self.config, n_freq)  # 변경: old algorithm에는 3채널 dB view만 제공


def _surrogate_normalize_targets(self, target_db: np.ndarray) -> np.ndarray:
    target_db = np.asarray(target_db, dtype=np.float32)
    n_freq = len(self.frequencies)
    db_view = _extract_db_view(target_db, self.config, n_freq)  # 변경: GUI 8채널 target을 old ResNet용 3채널로 축소
    outputs = []
    for idx, key in enumerate(["s11_db", "s21_db", "s22_db"]):
        chunk = db_view[idx * n_freq:(idx + 1) * n_freq]
        params = self.normalization.get(key, {})
        mode = self._norm_mode(key)
        if mode == "standard":
            clip = float(params.get("clip", 3.0))
            z = (chunk - float(params["mean"])) / (float(params["std"]) + 1e-8)
            z = np.clip(z, -clip, clip)
            outputs.append((z + clip) / (2.0 * clip))
        elif "s11_db" in self.normalization:
            lo = float(params.get("min", -140.0))
            hi = float(params.get("max", 0.0))
            outputs.append((chunk - lo) / (hi - lo + 1e-8))
        elif "s11_min" in self.normalization:
            lo = float(self.normalization[key.replace("_db", "_min")])
            hi = float(self.normalization[key.replace("_db", "_max")])
            outputs.append((chunk - lo) / (hi - lo + 1e-8))
        else:
            lo = float(self.normalization.get("s_min", -140.0))
            hi = float(self.normalization.get("s_max", 0.0))
            outputs.append((chunk - lo) / (hi - lo + 1e-8))
    return np.concatenate(outputs).astype(np.float32)


def _surrogate_predict(self, layouts: np.ndarray, return_std: bool = True):
    layouts = np.asarray(layouts, dtype=np.float32)
    if layouts.ndim == 2:
        layouts = layouts[None, ...]
    stacked = []
    with torch.no_grad():
        for model in self.models:
            chunks = []
            for start in range(0, len(layouts), self.chunk_size):
                end = min(start + self.chunk_size, len(layouts))
                batch = torch.from_numpy(layouts[start:end]).unsqueeze(1).to(self.device)
                chunks.append(model(batch).detach().cpu().numpy())
            stacked.append(np.concatenate(chunks, axis=0))
    stacked = np.stack(stacked, axis=0)
    mean_full = stacked.mean(axis=0)
    mean_raw = np.stack([_denormalize_full_prediction(self.config, self.frequencies, row) for row in mean_full], axis=0)
    n_freq = len(self.frequencies)
    mean = np.stack([_extract_db_view(row, self.config, n_freq) for row in mean_full], axis=0)  # 변경: old algorithm에는 3채널 mean만 반환
    std = np.stack([_extract_db_view(row, self.config, n_freq) for row in std_full], axis=0) if return_std else np.zeros_like(mean)
    return mean.astype(np.float32), std.astype(np.float32)


_GUIEnsembleSurrogate.denormalize_predictions = _surrogate_denormalize_predictions  # 변경: 8채널 모델 -> 3채널 old ResNet bridge 적용
_GUIEnsembleSurrogate.normalize_targets = _surrogate_normalize_targets  # 변경: target도 같은 브리지 적용
_GUIEnsembleSurrogate.predict = _surrogate_predict  # 변경: old algorithm이 3채널 출력을 받도록 보장


def _surrogate_predict_full_denorm(self, layouts: np.ndarray, return_std: bool = True):
    layouts = np.asarray(layouts, dtype=np.float32)
    if layouts.ndim == 2:
        layouts = layouts[None, ...]
    stacked = []
    with torch.no_grad():
        for model in self.models:
            chunks = []
            for start in range(0, len(layouts), self.chunk_size):
                end = min(start + self.chunk_size, len(layouts))
                batch = torch.from_numpy(layouts[start:end]).unsqueeze(1).to(self.device)
                chunks.append(model(batch).detach().cpu().numpy())
            stacked.append(np.concatenate(chunks, axis=0))
    mean_full = np.stack(stacked, axis=0).mean(axis=0)
    mean_raw = np.stack([_denormalize_full_prediction(self.config, self.frequencies, row) for row in mean_full], axis=0)
    n_freq = len(self.frequencies)
    mean = np.stack([_extract_db_view(row, self.config, n_freq) for row in mean_raw], axis=0)
    std = np.zeros_like(mean)
    return mean.astype(np.float32), std.astype(np.float32)


_GUIEnsembleSurrogate.predict = _surrogate_predict_full_denorm


def _build_target_spec(frequencies: np.ndarray, config: dict):
    """current GUI target을 old ResNet TargetSpec으로 변환."""  # 변경: 8채널 target에서도 dB 채널만 읽도록 재정의
    freqs = np.asarray(frequencies, dtype=np.float32)
    n_freq = len(freqs)
    target_db = np.asarray(config.get("targets_params", []), dtype=np.float32)
    if target_db.size == 0:
        target_db = np.concatenate([
            np.full_like(freqs, -10.0, dtype=np.float32),
            np.full_like(freqs, -3.0, dtype=np.float32),
            np.full_like(freqs, -10.0, dtype=np.float32),
        ])
    db_view = _extract_db_view(target_db, config, n_freq)
    pb_mask = (freqs >= float(config.get("f_low", freqs[0]))) & (freqs <= float(config.get("f_high", freqs[-1])))
    t11 = db_view[:n_freq]
    t21 = db_view[n_freq:2 * n_freq]
    t22 = db_view[2 * n_freq:3 * n_freq]
    il = float(np.mean(t21[pb_mask])) if np.any(pb_mask) else -3.0
    rl_vals = np.concatenate([t11[pb_mask], t22[pb_mask]]) if np.any(pb_mask) else np.array([-10.0], dtype=np.float32)
    rl = float(np.mean(rl_vals))
    sb_mask = ~pb_mask
    sb_attn = float(np.mean(t21[sb_mask])) if np.any(sb_mask) else il
    return _resnet_algo.TargetSpec(
        float(config.get("f_low", freqs[0])),
        float(config.get("f_high", freqs[-1])),
        il,
        rl,
        sb_attn,
    )


_S21_COMPLEX_PRIORITY_WEIGHT = 700.0    # S21 독주 방지 (절제된 가중치)
_S11_PRIORITY_WEIGHT         = 1800.0   # ★ S11 정합 (1500→1800 상향)
_S22_PRIORITY_WEIGHT         = 1500.0   # ★ S22 정합 (1100→1500 상향)
_S12_COMPLEX_PRIORITY_WEIGHT = 350.0
# ── 정합(Matching) 특화 페널티 가중치 (황금 밸런스 v3) ──────────────────────────
_S11_ORIGIN_PULL_WEIGHT      = 1500.0   # Γ² 강한 당김
_S22_ORIGIN_PULL_WEIGHT      = 1100.0   # Γ² 강한 당김
_MAX_REFL_PENALTY_WEIGHT     = 2000.0   # 최악 반사 강타 (Max Reflection Squared)
_VSWR_CENTER_WEIGHT          = 1400.0   # ★ VSWR 중심 당김 (1300→1400 상향)
_TV_LOSS_WEIGHT              = 0.2      # ★ 신규: TV Loss (공간적 연속성 보조, 0.1~0.3 추천)
_TX_RECIPROCITY_WEIGHT       = 120.0
_TX_PASSBAND_WEIGHT      = 10.0  # 삽입손실 가중치 하향 (S21 bias 억제)
_TX_STOPBAND_WEIGHT      = 20.0  # 저지대역 감쇠 (날카로운 전이구간 확보)
_TX_LOW_STOPBAND_MEAN_WEIGHT = 55.0
_TX_LOW_STOPBAND_PEAK_WEIGHT = 30.0
_LOW_FREQ_REFLECTION_WEIGHT  = 18.0
_LOW_FREQ_REFLECTION_LIMIT_DB = -3.0
_TX_TRANSITION_WEIGHT    = 35.0  # 전이구간 롤오프 선명도 (통과대역 외곽 근방 추가 패널티)
_TX_LF_OSC_WEIGHT        = 18.0  # 저주파 저지대역 발진 억제 (std 패널티)
_TX_RIPPLE_WEIGHT        =  2.0  # 통과대역 평탄도 (리플 억제)
_TX_COMPLEX_WEIGHT       = 60.0  # Polar S21/S12 매칭 - 복소 L2 거리로 크기+위상 선형성 동시 강제


def _infer_vector_keys(values: np.ndarray, n_freq: int) -> list[str]:
    """벡터 길이로부터 채널 구성을 추정한다."""  # 변경: retrieval/예측 데이터가 3채널/8채널 어느 쪽이든 자동 대응
    channel_count = int(np.asarray(values).shape[-1] // max(n_freq, 1))
    if channel_count == len(DEFAULT_TARGET_KEYS):
        return list(DEFAULT_TARGET_KEYS)
    if channel_count == 4:
        return ["s11_db", "s12_db", "s21_db", "s22_db"]
    return ["s11_db", "s21_db", "s22_db"]


def _denormalize_full_prediction(config: dict, frequencies: np.ndarray, pred_norm: np.ndarray) -> np.ndarray:
    """정규화된 전체 출력을 실제 스케일로 복원한다."""  # 변경: 8채널 전체 복원을 지원
    pred_norm = np.asarray(pred_norm, dtype=np.float32)
    n_freq = len(frequencies)
    target_keys = get_target_keys(config)
    normalization = config.get("normalization", {})
    normalization_type = config.get("normalization_type", "minmax")
    restored = []
    for key in target_keys:
        sl = channel_slice(target_keys, key, n_freq)
        chunk = pred_norm[sl]
        params = normalization.get(key, {})
        mode = params.get("type", normalization_type if "s11_db" in normalization else "minmax")
        if mode == "standard_direct":
            raw = chunk * float(params["std"]) + float(params["mean"])
            restored.append(raw)
        elif mode == "standard":
            clip = float(params.get("clip", 3.0))
            z = np.clip(chunk, 0.0, 1.0) * (2.0 * clip) - clip
            raw = z * float(params["std"]) + float(params["mean"])
            restored.append(np.clip(raw, float(params.get("raw_min", -200.0)), 0.0) if key.endswith("_db") else raw)
        elif "s11_db" in normalization:
            lo = float(params.get("min", -140.0))
            hi = float(params.get("max", 0.0 if key.endswith("_db") else 180.0))
            raw = np.clip(chunk, 0.0, 1.0) * (hi - lo) + lo
            restored.append(np.clip(raw, lo, 0.0) if key.endswith("_db") else raw)
        elif key.endswith("_db") and "s11_min" in normalization:
            lo = float(normalization[key.replace("_db", "_min")])
            hi = float(normalization[key.replace("_db", "_max")])
            raw = np.clip(chunk, 0.0, 1.0) * (hi - lo) + lo
            restored.append(np.clip(raw, lo, 0.0))
        elif key.endswith("_deg") and "s11_deg_min" in normalization:
            lo = float(normalization[key.replace("_deg", "_min")])
            hi = float(normalization[key.replace("_deg", "_max")])
            restored.append(np.clip(chunk, 0.0, 1.0) * (hi - lo) + lo)
        else:
            lo = float(normalization.get("s_min", -140.0))
            hi = float(normalization.get("s_max", 0.0))
            raw = np.clip(chunk, 0.0, 1.0) * (hi - lo) + lo
            restored.append(np.clip(raw, lo, 0.0) if key.endswith("_db") else raw)
    return np.concatenate(restored).astype(np.float32)


def _predict_full_stats(surrogate: _GUIEnsembleSurrogate, layouts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """앙상블 전체의 full output mean/std를 얻는다."""  # 변경: old surrogate.predict가 3채널로 축소하기 전에 full prediction 확보
    layouts = np.asarray(layouts, dtype=np.float32)
    if layouts.ndim == 2:
        layouts = layouts[None, ...]
    stacked = []
    with torch.no_grad():
        for model in surrogate.models:
            chunks = []
            for start in range(0, len(layouts), surrogate.chunk_size):
                end = min(start + surrogate.chunk_size, len(layouts))
                batch = torch.from_numpy(layouts[start:end]).unsqueeze(1).to(surrogate.device)
                chunks.append(model(batch).detach().cpu().numpy())
            stacked.append(np.concatenate(chunks, axis=0))
    stacked = np.stack(stacked, axis=0)
    return stacked.mean(axis=0).astype(np.float32), stacked.std(axis=0).astype(np.float32)


def _renormalize_complex_eval(s_complex: dict[str, np.ndarray], z01_new: complex, z02_new: complex, z0_old: float = 50.0) -> dict[str, np.ndarray]:
    """복소 S를 사용해 power-wave 재정규화한다."""  # 변경: GUI와 같은 복소 renorm으로 constraint 평가 정밀도 확보
    n_f = len(next(iter(s_complex.values())))

    def _gamma(z_new):
        return (z_new - z0_old) / (z_new + z0_old)

    def _d_factor(z_new):
        mag = np.sqrt(max(z_new.real, 1e-12) / z0_old)
        denom = abs(z0_old + z_new)
        corr = (z0_old + z_new) / denom if denom > 1e-15 else 1.0
        return complex(mag) * corr

    g1, g2 = _gamma(z01_new), _gamma(z02_new)
    d1, d2 = _d_factor(z01_new), _d_factor(z02_new)
    G = np.diag([g1, g2])
    Gc = np.conj(G)
    D = np.diag([d1, d2])
    Di = np.diag([1.0 / d1, 1.0 / d2])
    s12_src = s_complex.get("s12", s_complex.get("s21"))
    s11_new = np.zeros(n_f, dtype=complex)
    s12_new = np.zeros(n_f, dtype=complex)
    s21_new = np.zeros(n_f, dtype=complex)
    s22_new = np.zeros(n_f, dtype=complex)
    for i in range(n_f):
        S = np.array([[s_complex["s11"][i], s12_src[i]], [s_complex["s21"][i], s_complex["s22"][i]]], dtype=complex)
        A = S - Gc
        B = np.eye(2, dtype=complex) - G @ S
        S_new = D @ A @ np.linalg.solve(B, np.eye(2, dtype=complex)) @ Di
        s11_new[i] = S_new[0, 0]
        s12_new[i] = S_new[0, 1]
        s21_new[i] = S_new[1, 0]
        s22_new[i] = S_new[1, 1]

    to_db = lambda arr: 20.0 * np.log10(np.abs(arr) + 1e-15)
    return {
        "s11_db": np.clip(to_db(s11_new), a_min=None, a_max=0.0),
        "s12_db": np.clip(to_db(s12_new), a_min=None, a_max=0.0),
        "s21_db": np.clip(to_db(s21_new), a_min=None, a_max=0.0),
        "s22_db": np.clip(to_db(s22_new), a_min=None, a_max=0.0),
        "s11_complex": s11_new,
        "s22_complex": s22_new,
        "s21_complex": s21_new,  # 변경: (A) Polar S21 매칭을 위해 복소 S21 노출
        "s12_complex": s12_new,  # 변경: (A) Polar S12 매칭을 위해 복소 S12 노출
    }


def _prediction_eval_channels(values: np.ndarray, config: dict, frequencies: np.ndarray) -> dict[str, np.ndarray]:
    """예측 벡터를 실제 평가용 S11/S12/S21/S22 채널로 해석한다."""  # 변경: 8채널이면 복소 renorm, 3채널이면 dB-only fallback
    values = np.asarray(values, dtype=np.float32)
    n_freq = len(frequencies)
    target_keys = get_target_keys(config)
    if values.shape[-1] != len(target_keys) * n_freq:
        target_keys = _infer_vector_keys(values, n_freq)
    s_complex = complex_sparams_from_db_deg(values, target_keys, n_freq)
    z01_new = config.get("p1_z0", 50.0 + 0j)
    z02_new = config.get("p2_z0", 50.0 + 0j)
    if {"s11", "s21", "s22"}.issubset(s_complex.keys()):
        return _renormalize_complex_eval(s_complex, z01_new, z02_new)
    ch = split_prediction_channels(values, target_keys, n_freq)
    real_imag_complex = {}
    for port in ("s11", "s12", "s21", "s22"):
        rk = f"{port}_real"
        ik = f"{port}_imag"
        if rk in ch and ik in ch:
            real_imag_complex[port] = ch[rk] + 1j * ch[ik]
    if {"s11", "s21", "s22"}.issubset(real_imag_complex.keys()):
        return _renormalize_complex_eval(real_imag_complex, z01_new, z02_new)
    r_s11, r_s21, r_s22 = numpy_renormalize_s(ch["s11_db"], ch["s21_db"], ch["s22_db"], z0_old=50.0, z01_new=z01_new, z02_new=z02_new)
    return {
        "s11_db": np.clip(r_s11, a_min=None, a_max=0.0),
        "s12_db": np.clip(r_s21, a_min=None, a_max=0.0),
        "s21_db": np.clip(r_s21, a_min=None, a_max=0.0),
        "s22_db": np.clip(r_s22, a_min=None, a_max=0.0),
        "s11_complex": np.zeros_like(r_s11, dtype=complex),
        "s22_complex": np.zeros_like(r_s22, dtype=complex),
        "s21_complex": np.zeros_like(r_s21, dtype=complex),  # 변경: (A) 3채널 폴백에서도 키 일관성 유지
        "s12_complex": np.zeros_like(r_s21, dtype=complex),  # 변경: (A) 3채널 폴백에서도 키 일관성 유지
    }


def _split_line_bounds(index: int, width: int, limit: int) -> tuple[int, int]:
    """분리선 폭을 중심 index 기준의 안전한 정수 구간으로 변환한다."""  # 변경: 분리선 좌표 계산을 공통 함수로 정리
    width = max(1, int(width))
    index = int(index)
    half_left = (width - 1) // 2
    half_right = width // 2
    start = max(0, index - half_left)
    stop = min(limit, index + half_right + 1)
    return start, stop


def _split_line_enabled(config: dict | None) -> bool:
    """설정에서 분리선 사용 여부를 읽는다."""  # 변경: 분리선 옵션 파싱을 단순화
    cfg = config or {}
    mode = str(cfg.get("split_line_mode", "none")).strip().lower()
    return bool(cfg.get("split_line_enabled", False)) and mode not in {"", "none", "off", "disable", "disabled"}


def _port_regions_connected_8(layout: np.ndarray) -> bool:
    """포트1/포트2가 8-이웃 기준으로 연결됐는지 검사한다."""  # 변경: 대각선도 연결로 보는 실제 포트 분리 검사를 추가
    grid = np.asarray(layout, dtype=np.float32) >= 0.5
    if grid.ndim != 2:
        return False

    left_mask = np.zeros_like(grid, dtype=bool)
    right_mask = np.zeros_like(grid, dtype=bool)
    left_mask[11:14, :2] = True
    right_mask[11:14, 23:] = True

    start_cells = np.argwhere(grid & left_mask)
    target_mask = grid & right_mask
    if start_cells.size == 0 or not np.any(target_mask):
        return False

    visited = np.zeros_like(grid, dtype=bool)
    stack = [tuple(cell) for cell in start_cells]
    for r, c in stack:
        visited[r, c] = True

    neighbors = [  # 변경: 대각선까지 포함한 8-이웃 연결 기준 사용
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    height, width = grid.shape
    while stack:
        r, c = stack.pop()
        if target_mask[r, c]:
            return True
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            if visited[nr, nc] or not grid[nr, nc]:
                continue
            visited[nr, nc] = True
            stack.append((nr, nc))
    return False


def _jagged_centers(length: int, index: int, limit: int, seed: int) -> np.ndarray:
    """실행마다 고정되지만 직선만은 아닌 지그재그 중심선을 만든다."""  # 변경: 대각선이 섞인 랜덤 분리선 경로 생성
    rng = np.random.default_rng(int(seed))
    lo = 0
    hi = max(0, limit - 1)
    pos = int(np.clip(index, lo, hi))
    centers = np.empty(length, dtype=np.int32)
    for i in range(length):
        centers[i] = pos
        step = int(rng.choice([-1, 0, 1], p=[0.30, 0.40, 0.30]))  # 변경: 한 칸 폭을 유지하기 위해 한 번에 한 칸만 좌우로 이동
        pos = int(np.clip(pos + step, lo, hi))
    return centers


def _apply_jagged_vertical(arr: np.ndarray, index: int, width: int, fill_value: float, seed: int) -> None:
    centers = _jagged_centers(arr.shape[-2], index, arr.shape[-1], seed)  # 변경: 각 행마다 조금씩 다른 열 중심을 사용
    prev_center = None
    for row, col_center in enumerate(centers):
        c0, c1 = _split_line_bounds(int(col_center), width, arr.shape[-1])
        arr[:, row, c0:c1] = fill_value
        if prev_center is not None and int(col_center) != int(prev_center):
            bridge0 = min(int(prev_center), int(col_center))
            bridge1 = max(int(prev_center), int(col_center)) + 1
            arr[:, row, bridge0:bridge1] = fill_value
        prev_center = int(col_center)


def _apply_jagged_horizontal(arr: np.ndarray, index: int, width: int, fill_value: float, seed: int) -> None:
    centers = _jagged_centers(arr.shape[-1], index, arr.shape[-2], seed)  # 변경: 각 열마다 조금씩 다른 행 중심을 사용
    prev_center = None
    for col, row_center in enumerate(centers):
        r0, r1 = _split_line_bounds(int(row_center), width, arr.shape[-2])
        arr[:, r0:r1, col] = fill_value
        if prev_center is not None and int(row_center) != int(prev_center):
            bridge0 = min(int(prev_center), int(row_center))
            bridge1 = max(int(prev_center), int(row_center)) + 1
            arr[:, bridge0:bridge1, col] = fill_value
        prev_center = int(row_center)


def _apply_split_line(layouts: np.ndarray | None, config: dict | None, fill_value: float = 0.0) -> np.ndarray | None:
    """레이아웃에 분리선을 강제로 삽입해 포트 간 DC 경로를 끊는다."""  # 변경: 평가/표시/반환 모두 같은 분리선 형상을 사용
    if layouts is None:
        return None
    arr = np.asarray(layouts, dtype=np.float32).copy()
    if not _split_line_enabled(config):
        return arr

    cfg = config or {}
    mode = str(cfg.get("split_line_mode", "vertical")).strip().lower()
    width = max(1, int(cfg.get("split_line_width", 1)))
    index = int(cfg.get("split_line_index", GRID_SIZE // 2))
    pattern = str(cfg.get("split_line_pattern", "straight")).strip().lower()  # 변경: 직선/지그재그 분리선 패턴을 설정에서 읽음
    seed = int(cfg.get("split_line_seed", 0))  # 변경: 한 번의 실행 안에서는 같은 랜덤 분리선이 재현되도록 seed 사용
    fill_value = float(fill_value)

    if arr.ndim == 2:
        arr = arr[None, ...]
        squeeze = True
    else:
        squeeze = False

    if mode in {"vertical", "cross"}:
        if pattern == "jagged":
            _apply_jagged_vertical(arr, index, width, fill_value, seed + 17)  # 변경: 세로 분리선은 지그재그/대각선 경로로 삽입
        else:
            c0, c1 = _split_line_bounds(index, width, arr.shape[-1])
            arr[:, :, c0:c1] = fill_value
    if mode in {"horizontal", "cross"}:
        if pattern == "jagged":
            _apply_jagged_horizontal(arr, index, width, fill_value, seed + 31)  # 변경: 가로 분리선도 동일하게 지그재그 경로 지원
        else:
            r0, r1 = _split_line_bounds(index, width, arr.shape[-2])
            arr[:, r0:r1, :] = fill_value

    return arr[0] if squeeze else arr


def _split_line_mask(config: dict | None) -> np.ndarray:
    """현재 분리선 설정을 25x25 금지영역 마스크로 변환한다."""  # 변경: 분리선을 후처리가 아닌 실제 금지영역으로 쓰기 위한 마스크 추가
    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    if not _split_line_enabled(config):
        return mask
    base = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    carved = _apply_split_line(base, config, fill_value=0.0)
    return np.asarray(carved, dtype=np.float32) < 0.5


def _enforce_split_mask(layouts: np.ndarray | None, config: dict | None) -> np.ndarray | None:
    """분리선 금지영역에는 픽셀이 절대 생기지 않도록 0으로 고정한다."""  # 변경: 탐색 과정 전체에서 분리선 셀을 영구적으로 비우기 위한 clamp 추가
    if layouts is None:
        return None
    arr = np.asarray(layouts, dtype=np.float32).copy()
    mask = _split_line_mask(config)
    if not np.any(mask):
        return arr
    if arr.ndim == 2:
        arr[mask] = 0.0
    elif arr.ndim == 3:
        arr[:, mask] = 0.0
    else:
        raise ValueError(f"Unsupported layout rank for split mask enforcement: {arr.ndim}")
    return arr


def _split_mask_tensor(device: torch.device, dtype: torch.dtype, config: dict | None) -> torch.Tensor | None:
    """torch 연산에서 재사용할 분리선 금지영역 마스크를 만든다."""  # 변경: GD 단계에서도 분리선 셀을 0으로 강제하기 위한 torch 마스크 추가
    mask_np = _split_line_mask(config)
    if not np.any(mask_np):
        return None
    return torch.from_numpy(mask_np).to(device=device, dtype=dtype).view(1, 1, GRID_SIZE, GRID_SIZE)


def _stabilize_split_line_config(config: dict | None, attempts: int = 64) -> dict | None:
    """포트를 실제로 끊는 분리선이 나올 때까지 재샘플링한다."""  # 변경: 벽에 붙거나 포트를 못 끊는 분리선은 시작 전에 다시 생성
    if config is None:
        return None
    cfg = dict(config)
    if not _split_line_enabled(cfg):
        return cfg

    mode = str(cfg.get("split_line_mode", "vertical")).strip().lower()
    width = max(1, int(cfg.get("split_line_width", 1)))
    rng = np.random.default_rng(int(cfg.get("split_line_seed", 0)) + 7919)

    def _random_index() -> int:
        if mode in {"vertical", "cross"}:
            lo = max(3, width + 2)
            hi = min(GRID_SIZE - 4, GRID_SIZE - width - 3)
        else:
            lo = max(3, width + 2)
            hi = min(GRID_SIZE - 4, GRID_SIZE - width - 3)
        if hi < lo:
            return GRID_SIZE // 2
        return int(rng.integers(lo, hi + 1))

    candidate = dict(cfg)
    for _ in range(max(1, int(attempts))):
        candidate["split_line_index"] = _random_index()
        candidate["split_line_seed"] = int(rng.integers(1, 1_000_000_000))
        fully_open = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        carved = _enforce_split_mask(fully_open, candidate)
        if not _port_regions_connected_8(carved):
            return candidate
    return cfg


def _constraint_first_score(
    eval_channels: dict[str, np.ndarray],
    frequencies: np.ndarray,
    spec,
    layout: np.ndarray | None = None,
    config: dict | None = None,
) -> float:
    """반사계수 제약 우선 + 날카로운 전이구간 + 저주파 발진 억제 점수."""
    freqs = np.asarray(frequencies, dtype=np.float32)
    f_low, f_high = float(spec.f_low), float(spec.f_high)

    pb_mask  = (freqs >= f_low) & (freqs <= f_high)
    # 전이구간 마스크: 통과대역 외곽 10% 이내
    tr_ratio = 0.10
    tr_low   = f_low  * (1.0 - tr_ratio)
    tr_high  = f_high * (1.0 + tr_ratio)
    trans_mask = (
        ((freqs >= tr_low) & (freqs < f_low)) |
        ((freqs > f_high) & (freqs <= tr_high))
    )
    sb_mask  = ~(pb_mask | trans_mask)
    low_stop_mask = freqs < tr_low
    # 저주파 저지대역 전체: 발진과 누설을 따로 강하게 억제
    lf_mask  = low_stop_mask

    # ── 반사계수 (S11/S22) 제약 ──────────────────────────────────────────
    gamma_limit = 10.0 ** (float(spec.return_loss) / 20.0)
    gamma11 = np.abs(eval_channels.get("s11_complex", np.zeros_like(freqs, dtype=complex)))
    gamma22 = np.abs(eval_channels.get("s22_complex", np.zeros_like(freqs, dtype=complex)))
    if not np.any(gamma11):
        gamma11 = 10.0 ** (eval_channels["s11_db"] / 20.0)
    if not np.any(gamma22):
        gamma22 = 10.0 ** (eval_channels["s22_db"] / 20.0)
    # === 1. 정합 관련 강력 페널티 (최종 추천안 적용) ===
    # 힌지 로스를 제곱(Squared)하여 경계를 넘었을 때 더욱 강력하게 타격
    refl11_penalty = np.mean(np.maximum(0.0, gamma11[pb_mask] - gamma_limit)**2) if np.any(pb_mask) else 0.0
    refl22_penalty = np.mean(np.maximum(0.0, gamma22[pb_mask] - gamma_limit)**2) if np.any(pb_mask) else 0.0

    # Max Reflection Penalty (양쪽 중 더 나쁜 쪽 강타)
    max_refl = np.maximum(gamma11[pb_mask], gamma22[pb_mask]) if np.any(pb_mask) else np.array([0.0])
    max_refl_penalty = float(np.mean(np.maximum(0.0, max_refl - gamma_limit)**2))

    # VSWR Center Pull (가장 강력한 중심 당김)
    center_loss = float(np.mean(gamma11[pb_mask]**2 + gamma22[pb_mask]**2)) if np.any(pb_mask) else 0.0

    # Origin Pull (이미 renorm된 Γ를 원점으로 2차 당김)
    origin_pull11 = float(np.mean(gamma11[pb_mask] ** 2)) if np.any(pb_mask) else 0.0
    origin_pull22 = float(np.mean(gamma22[pb_mask] ** 2)) if np.any(pb_mask) else 0.0

    s11_db = eval_channels["s11_db"]
    s22_db = eval_channels["s22_db"]
    s21 = eval_channels["s21_db"]
    s12 = eval_channels.get("s12_db", s21)
    s21_complex = np.asarray(eval_channels.get("s21_complex", np.zeros_like(freqs, dtype=complex)), dtype=np.complex64)  # 변경: 1단계만 적용, 복소 S21 패널티 계산용 채널 확보
    s12_complex = np.asarray(eval_channels.get("s12_complex", np.zeros_like(freqs, dtype=complex)), dtype=np.complex64)  # 변경: 1단계만 적용, 복소 S12 패널티 계산용 채널 확보
    if not np.any(np.abs(s21_complex)):
        s21_complex = (10.0 ** (s21 / 20.0)).astype(np.complex64)  # 변경: 3채널 폴백 모델에서도 선형 크기 기반 복소 패널티를 계산
    if not np.any(np.abs(s12_complex)):
        s12_complex = (10.0 ** (s12 / 20.0)).astype(np.complex64)  # 변경: 3채널 폴백 모델에서도 선형 크기 기반 복소 패널티를 계산

    s12_complex_penalty = 0.0  # 복소 S12 Polar L2
    s21_complex_penalty = 0.0  # 복소 S21 Polar L2
    reciprocity_penalty = 0.0  # S12/S21 일치도 (Reciprocity)
    if np.any(pb_mask):
        # 변경: 통과대역 복소 타겟을 "위상 적응형"으로 구성
        # 필터는 물리적으로 통과대역에서 위상이 선형으로 감소(일정 그룹지연)하므로
        # 목표를 순수 실수로 두면 옵티마이저가 물리와 싸우게 됨 → 예측 위상에 선형 피팅해 그 선형 위상을 타겟 위상으로 사용
        target_mag_pb = 10.0 ** (float(spec.insertion_loss) / 20.0)  # 통과대역 목표 |S| (선형 스케일)
        pb_indices = np.nonzero(pb_mask)[0]
        f_pb = freqs[pb_mask]

        def _polar_penalty(s_complex: np.ndarray) -> float:
            """예측 위상에 선형 피팅한 후 |S_pred - target_mag * exp(j * linear_phase)|^2 평균."""
            vals = s_complex[pb_mask]
            if f_pb.size >= 2:
                phi = np.unwrap(np.angle(vals + 1e-15))
                slope, intercept = np.polyfit(f_pb.astype(np.float64), phi.astype(np.float64), 1)
                target_phase = slope * f_pb + intercept
            else:
                target_phase = np.angle(vals + 1e-15)
            target = target_mag_pb * np.exp(1j * target_phase)
            return float(np.mean(np.abs(vals - target) ** 2))

        s12_complex_penalty = _polar_penalty(s12_complex)  # 변경: S12 Polar L2 (위상 적응)
        s21_complex_penalty = _polar_penalty(s21_complex)  # 변경: S21 Polar L2 (위상 적응)
        reciprocity_penalty = float(np.mean(np.abs(s12_complex[pb_mask] - s21_complex[pb_mask]) ** 2))  # 상호회로 가정

    # ── 통과대역 삽입손실 패널티 ─────────────────────────────────────────
    tx_pb_penalty = (
        np.maximum(0.0, float(spec.insertion_loss) - s21[pb_mask]).mean()
        + np.maximum(0.0, float(spec.insertion_loss) - s12[pb_mask]).mean()
    ) if np.any(pb_mask) else 0.0

    # ── 저지대역 감쇠 패널티 ─────────────────────────────────────────────
    tx_sb_penalty = (
        np.maximum(0.0, s21[sb_mask] - float(spec.stopband_attn)).mean()
        + np.maximum(0.0, s12[sb_mask] - float(spec.stopband_attn)).mean()
    ) if np.any(sb_mask) else 0.0

    low_tx_mean_penalty = 0.0
    low_tx_peak_penalty = 0.0
    low_reflection_penalty = 0.0
    if np.any(low_stop_mask):
        low_s21_leak = np.maximum(0.0, s21[low_stop_mask] - float(spec.stopband_attn))
        low_s12_leak = np.maximum(0.0, s12[low_stop_mask] - float(spec.stopband_attn))
        low_tx_mean_penalty = float(np.mean(low_s21_leak ** 2) + np.mean(low_s12_leak ** 2))
        low_tx_peak_penalty = float(max(np.max(low_s21_leak ** 2), np.max(low_s12_leak ** 2)))

        # Stopband should mostly reflect. A deep S11/S22 dip at low frequency is accidental matching.
        low_s11_match = np.maximum(0.0, _LOW_FREQ_REFLECTION_LIMIT_DB - s11_db[low_stop_mask])
        low_s22_match = np.maximum(0.0, _LOW_FREQ_REFLECTION_LIMIT_DB - s22_db[low_stop_mask])
        low_reflection_penalty = float(np.mean(low_s11_match ** 2) + np.mean(low_s22_match ** 2))

    # ── 전이구간 날카로움 패널티 ─────────────────────────────────────────
    # 전이구간에서 IL~SB 중간값보다 삽입손실이 높으면 페널티 (급격한 롤오프 유도)
    if np.any(trans_mask):
        tr_target = (float(spec.insertion_loss) + float(spec.stopband_attn)) / 2.0
        tx_tr_penalty = (
            np.maximum(0.0, s21[trans_mask] - tr_target).mean()
            + np.maximum(0.0, s12[trans_mask] - tr_target).mean()
        )
    else:
        tx_tr_penalty = 0.0

    # ── 저주파 발진 억제 패널티 (저지대역 내 S21/S12 표준편차) ─────────
    lf_osc_penalty = 0.0
    if np.any(lf_mask):
        lf_osc_penalty = float(np.std(s21[lf_mask]) + np.std(s12[lf_mask]))

    # ── 통과대역 리플 패널티 ─────────────────────────────────────────────
    ripple_penalty = (
        np.std(s21[pb_mask]) + np.std(s12[pb_mask])
    ) if np.any(pb_mask) else 0.0

    # ── 레이아웃 물리 제약 ───────────────────────────────────────────────
    layout_penalty = float(_resnet_algo.layout_penalty(layout)[0]) if layout is not None else 0.0
    connectivity_penalty = 0.0
    if layout is not None and _split_line_enabled(config):
        if _port_regions_connected_8(layout):
            connectivity_penalty = 1_000_000.0  # 변경: 분리선 모드에서 포트가 조금이라도 이어지면 사실상 즉시 탈락시키는 강한 페널티

    # ── TV Loss (공간적 연속성) ──────────────────────────────────────────
    tv_penalty = 0.0
    if layout is not None:
        diff_h = np.abs(layout[1:, :] - layout[:-1, :])
        diff_w = np.abs(layout[:, 1:] - layout[:, :-1])
        tv_penalty = float(np.mean(diff_h) + np.mean(diff_w))

    return float(
        _S21_COMPLEX_PRIORITY_WEIGHT * s21_complex_penalty
        + _S11_PRIORITY_WEIGHT        * refl11_penalty
        + _S22_PRIORITY_WEIGHT        * refl22_penalty
        + _MAX_REFL_PENALTY_WEIGHT    * max_refl_penalty  # 핵심: 최악 반사 강타 ★
        + _VSWR_CENTER_WEIGHT         * center_loss       # 핵심: VSWR 중심 당김 ★
        + _S11_ORIGIN_PULL_WEIGHT     * origin_pull11
        + _S22_ORIGIN_PULL_WEIGHT     * origin_pull22
        + _TV_LOSS_WEIGHT             * tv_penalty        # 신규: TV Loss 보조 ★
        + _S12_COMPLEX_PRIORITY_WEIGHT * s12_complex_penalty
        + _TX_RECIPROCITY_WEIGHT      * reciprocity_penalty
        + _TX_PASSBAND_WEIGHT         * tx_pb_penalty
        + _TX_STOPBAND_WEIGHT         * tx_sb_penalty
        + _TX_LOW_STOPBAND_MEAN_WEIGHT * low_tx_mean_penalty
        + _TX_LOW_STOPBAND_PEAK_WEIGHT * low_tx_peak_penalty
        + _LOW_FREQ_REFLECTION_WEIGHT  * low_reflection_penalty
        + _TX_TRANSITION_WEIGHT       * tx_tr_penalty
        + _TX_LF_OSC_WEIGHT           * lf_osc_penalty
        + _TX_RIPPLE_WEIGHT           * ripple_penalty
        + layout_penalty
        + connectivity_penalty
    )

def _constraint_fitness_np(layouts: np.ndarray, surrogate: _GUIEnsembleSurrogate, target_db: np.ndarray, spec, args=None) -> np.ndarray:
    """old ResNet 알고리즘이 호출할 공통 fitness를 제약 우선 방식으로 교체한다."""  # 변경: GA/BPSO/GD/DBS 전체가 같은 constraint-first score 사용
    layouts = np.asarray(layouts, dtype=np.float32)
    if layouts.ndim == 2:
        layouts = layouts[None, ...]
    layouts_eval = _enforce_split_mask(layouts, surrogate.config)  # 변경: 모든 후보를 분리선 금지영역이 비워진 실제 탐색 형상으로 평가
    mean_full, _std_full = _predict_full_stats(surrogate, layouts_eval)
    pred_full = np.stack([_denormalize_full_prediction(surrogate.config, surrogate.frequencies, item) for item in mean_full], axis=0)
    scores = [
        _constraint_first_score(_prediction_eval_channels(pred_full[idx], surrogate.config, surrogate.frequencies), surrogate.frequencies, spec, layouts_eval[idx], surrogate.config)  # 변경: 분리선/포트 연결성 페널티까지 현재 GUI 설정 기준으로 함께 계산
        for idx in range(len(pred_full))
    ]
    return np.asarray(scores, dtype=np.float32)


def _constraint_retrieval_seed(target_db: np.ndarray, retrieval_layouts: np.ndarray, retrieval_targets: np.ndarray, freqs: np.ndarray, spec, topk: int = 16, count: int = 8) -> list[np.ndarray]:
    """retrieval seed도 같은 제약 우선 점수로 선택한다."""  # 변경: 시작점부터 Smith-chart 제약을 우선하는 후보 사용
    config = {"target_keys": _infer_vector_keys(retrieval_targets, len(freqs)), "p1_z0": 50.0 + 0j, "p2_z0": 50.0 + 0j}
    split_cfg = _ACTIVE_SPLIT_LINE_CONFIG  # 변경: 현재 GUI 분리선 설정을 retrieval seed 선택에도 반영
    scores = []
    for idx in range(len(retrieval_layouts)):
        eval_channels = _prediction_eval_channels(retrieval_targets[idx], config, freqs)
        candidate_layout = _enforce_split_mask(retrieval_layouts[idx], split_cfg)  # 변경: retrieval 후보도 분리선 금지영역이 비워진 상태로 점수화
        scores.append(_constraint_first_score(eval_channels, freqs, spec, candidate_layout, split_cfg))
    top_idx = np.argsort(np.asarray(scores, dtype=np.float32))[:topk]
    chosen = top_idx[: min(count, len(top_idx))]
    return [_enforce_split_mask(retrieval_layouts[idx], split_cfg) for idx in chosen]  # 변경: 반환 seed 자체도 분리선 금지영역이 비워진 상태로 통일


_ORIG_RANDOM_POPULATION = _resnet_algo.random_population  # 변경: 분리선 금지영역 강제를 위해 원본 population 생성기를 보관
_ORIG_BUILD_SEEDED_POPULATION = _resnet_algo.build_seeded_population  # 변경: 원본 seeded population 생성기를 보관
_ORIG_MUTATE_POPULATION = _resnet_algo.mutate_population  # 변경: 원본 mutation을 보관
_ORIG_CROSSOVER_POPULATION = _resnet_algo.crossover_population  # 변경: 원본 crossover를 보관
_ORIG_RUN_BPSO = _resnet_algo.run_bpso  # 변경: 원본 BPSO를 보관
_ORIG_RUN_GD = _resnet_algo.run_gd  # 변경: 원본 GD를 보관
_ORIG_RUN_DBS = _resnet_algo.run_dbs  # 변경: 원본 DBS를 보관


def _split_aware_random_population(pop_size: int) -> np.ndarray:
    """분리선 금지영역을 비운 초기 population을 만든다."""  # 변경: 분리선 위에는 시작부터 픽셀이 생기지 않도록 초기화
    return _enforce_split_mask(_ORIG_RANDOM_POPULATION(pop_size), _ACTIVE_SPLIT_LINE_CONFIG)


def _split_aware_build_seeded_population(pop_size: int, seeds: list[np.ndarray]) -> np.ndarray:
    """seed를 포함한 초기 population에도 분리선 금지영역을 강제한다."""  # 변경: retrieval seed를 써도 분리선 셀은 항상 0으로 유지
    seeds = [_enforce_split_mask(seed, _ACTIVE_SPLIT_LINE_CONFIG) for seed in seeds]
    return _enforce_split_mask(_ORIG_BUILD_SEEDED_POPULATION(pop_size, seeds), _ACTIVE_SPLIT_LINE_CONFIG)


def _split_aware_mutate_population(population: np.ndarray, rate: float) -> np.ndarray:
    """mutation 후에도 분리선 금지영역에는 픽셀이 생기지 않도록 다시 clamp한다."""  # 변경: 변이 연산이 금지영역을 침범하지 못하도록 후처리
    return _enforce_split_mask(_ORIG_MUTATE_POPULATION(population, rate), _ACTIVE_SPLIT_LINE_CONFIG)


def _split_aware_crossover_population(population: np.ndarray, elite_count: int) -> np.ndarray:
    """crossover 후에도 분리선 금지영역에는 픽셀이 생기지 않도록 다시 clamp한다."""  # 변경: 교차 연산이 금지영역을 침범하지 못하도록 후처리
    return _enforce_split_mask(_ORIG_CROSSOVER_POPULATION(population, elite_count), _ACTIVE_SPLIT_LINE_CONFIG)


def _split_aware_run_bpso(surrogate, target_db, spec, seeds, particles, iterations, args, callback=None):
    """BPSO 위치 갱신 전체에 분리선 금지영역을 강제한다."""  # 변경: BPSO도 분리선 셀을 절대 1로 만들지 못하게 보강
    position = _split_aware_build_seeded_population(particles, seeds)
    velocity = np.zeros_like(position, dtype=np.float32)
    scores = _resnet_algo.fitness_np(position, surrogate, target_db, spec, args=args)
    pbest = position.copy()
    pbest_scores = scores.copy()
    best_idx = int(np.argmin(scores))
    gbest = position[best_idx].copy()
    gbest_score = float(scores[best_idx])

    for iteration in range(iterations):
        inertia = 0.9 - 0.5 * (iteration / max(iterations, 1))
        r1 = np.random.rand(*position.shape).astype(np.float32)
        r2 = np.random.rand(*position.shape).astype(np.float32)
        velocity = inertia * velocity + 2.0 * r1 * (pbest - position) + 2.0 * r2 * (gbest - position)
        probability = 1.0 / (1.0 + np.exp(-velocity))
        position = (np.random.rand(*position.shape) < probability).astype(np.float32)
        position = _enforce_split_mask(position, _ACTIVE_SPLIT_LINE_CONFIG)

        scores = _resnet_algo.fitness_np(position, surrogate, target_db, spec, args=args)
        improved = scores < pbest_scores
        pbest[improved] = position[improved]
        pbest = _enforce_split_mask(pbest, _ACTIVE_SPLIT_LINE_CONFIG)
        pbest_scores[improved] = scores[improved]

        best_idx = int(np.argmin(pbest_scores))
        if pbest_scores[best_idx] < gbest_score:
            gbest_score = float(pbest_scores[best_idx])
            gbest = pbest[best_idx].copy()
        gbest = _enforce_split_mask(gbest, _ACTIVE_SPLIT_LINE_CONFIG)
        if callback is not None and (iteration % 10 == 0 or iteration == iterations - 1):
            callback("bpso", iteration, gbest_score, gbest.copy(), None)
    return gbest, gbest_score


def _torch_denormalize_full_prediction(surrogate, pred_norm: torch.Tensor) -> torch.Tensor:
    config = surrogate.config or {}
    target_keys = get_target_keys(config)
    normalization = config.get("normalization", {})
    normalization_type = config.get("normalization_type", "minmax")
    n_freq = len(surrogate.frequencies)
    restored = []
    for key in target_keys:
        sl = channel_slice(target_keys, key, n_freq)
        chunk = pred_norm[:, sl]
        params = normalization.get(key, {})
        mode = params.get("type", normalization_type if "s11_db" in normalization else "minmax")
        if mode == "standard_direct":
            raw = chunk * float(params["std"]) + float(params["mean"])
        elif mode == "standard":
            clip = float(params.get("clip", 3.0))
            z = torch.clamp(chunk, 0.0, 1.0) * (2.0 * clip) - clip
            raw = z * float(params["std"]) + float(params["mean"])
            if key.endswith("_db"):
                raw = torch.clamp(raw, min=float(params.get("raw_min", -200.0)), max=0.0)
        elif "s11_db" in normalization:
            lo = float(params.get("min", -140.0))
            hi = float(params.get("max", 0.0 if key.endswith("_db") else 180.0))
            raw = torch.clamp(chunk, 0.0, 1.0) * (hi - lo) + lo
            if key.endswith("_db"):
                raw = torch.clamp(raw, min=lo, max=0.0)
        elif key.endswith("_db") and "s11_min" in normalization:
            lo = float(normalization[key.replace("_db", "_min")])
            hi = float(normalization[key.replace("_db", "_max")])
            raw = torch.clamp(chunk, 0.0, 1.0) * (hi - lo) + lo
            raw = torch.clamp(raw, min=lo, max=0.0)
        elif key.endswith("_deg") and "s11_deg_min" in normalization:
            lo = float(normalization[key.replace("_deg", "_min")])
            hi = float(normalization[key.replace("_deg", "_max")])
            raw = torch.clamp(chunk, 0.0, 1.0) * (hi - lo) + lo
        else:
            lo = float(normalization.get("s_min", -140.0))
            hi = float(normalization.get("s_max", 0.0))
            raw = torch.clamp(chunk, 0.0, 1.0) * (hi - lo) + lo
            if key.endswith("_db"):
                raw = torch.clamp(raw, min=lo, max=0.0)
        restored.append(raw)
    return torch.cat(restored, dim=1)


def _torch_extract_db_view(values_raw: torch.Tensor, config: dict, n_freq: int) -> torch.Tensor:
    target_keys = get_target_keys(config or {})
    key_to_chunk = {
        key: values_raw[:, channel_slice(target_keys, key, n_freq)]
        for key in target_keys
    }
    if {"s11_real", "s11_imag", "s21_real", "s21_imag", "s22_real", "s22_imag"}.issubset(key_to_chunk.keys()):
        out = []
        for port in ("s11", "s21", "s22"):
            real = key_to_chunk[f"{port}_real"]
            imag = key_to_chunk[f"{port}_imag"]
            mag = torch.sqrt(real * real + imag * imag + 1e-30)
            out.append(20.0 * torch.log10(torch.clamp(mag, min=1e-15)))
        return torch.cat(out, dim=1)
    return torch.cat([key_to_chunk[key] for key in ("s11_db", "s21_db", "s22_db")], dim=1)


def _split_aware_run_gd(surrogate, target_db, spec, init_layout, args, steps=_resnet_algo.GD_MAX_ITERS, lr=0.05, temp_start=2.0, temp_end=0.25, callback=None):
    """GD 확률 맵과 hard layout 모두에 분리선 금지영역을 강제한다."""  # 변경: GD 최적화 중에도 분리선 셀은 확률적으로조차 켜지지 않도록 보강
    cfg = _ACTIVE_SPLIT_LINE_CONFIG
    if init_layout is None:
        init_layout = _split_aware_random_population(1)[0]
    else:
        init_layout = _enforce_split_mask(init_layout, cfg)

    logits = torch.from_numpy((init_layout * 2.0 - 1.0) * 2.0).view(1, 1, GRID_SIZE, GRID_SIZE).to(surrogate.device)
    logits = logits.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=lr)

    mask_tensor = _split_mask_tensor(surrogate.device, logits.dtype, cfg)
    if mask_tensor is not None:
        with torch.no_grad():
            logits.data.masked_fill_(mask_tensor.bool(), -12.0)

    target_view = _extract_db_view(target_db, surrogate.config, len(surrogate.frequencies))
    target_tensor = torch.from_numpy(target_view).view(1, -1).to(surrogate.device)
    best_layout = init_layout.copy()
    best_score = float("inf")

    freqs_np = np.asarray(surrogate.frequencies, dtype=np.float32)
    passband = torch.from_numpy(((freqs_np >= spec.f_low) & (freqs_np <= spec.f_high)).astype(np.bool_)).to(surrogate.device)
    stopband = ~passband
    low_stopband = torch.from_numpy((freqs_np < float(spec.f_low) * 0.90).astype(np.bool_)).to(surrogate.device)
    n_freq = len(surrogate.frequencies)

    for step in range(steps):
        optimizer.zero_grad()
        temperature = temp_start + (temp_end - temp_start) * (step / max(steps - 1, 1))
        prob = torch.sigmoid(logits / max(temperature, 1e-4))
        hard = (prob > 0.5).float()
        x = hard + prob - prob.detach()
        if mask_tensor is not None:
            prob = prob.masked_fill(mask_tensor.bool(), 0.0)
            hard = hard.masked_fill(mask_tensor.bool(), 0.0)
            x = x.masked_fill(mask_tensor.bool(), 0.0)

        outputs = [model(x) for model in surrogate.models]
        pred_norm_full = torch.stack(outputs, dim=0).mean(dim=0)
        pred_raw_full = _torch_denormalize_full_prediction(surrogate, pred_norm_full)
        pred = _torch_extract_db_view(pred_raw_full, surrogate.config, n_freq)

        pred_s11, pred_s21, pred_s22 = pred[:, :n_freq], pred[:, n_freq:2 * n_freq], pred[:, 2 * n_freq:3 * n_freq]
        tgt_s11, tgt_s21, tgt_s22 = target_tensor[:, :n_freq], target_tensor[:, n_freq:2 * n_freq], target_tensor[:, 2 * n_freq:3 * n_freq]
        loss = (
            args.w_s21_passband * torch.relu(tgt_s21[:, passband] - pred_s21[:, passband]).mean()
            + args.w_s11_passband * torch.relu(pred_s11[:, passband] - tgt_s11[:, passband]).mean()
            + args.w_s22_passband * torch.relu(pred_s22[:, passband] - tgt_s22[:, passband]).mean()
            + args.w_s21_stopband * torch.relu(pred_s21[:, stopband] - tgt_s21[:, stopband]).mean()
        )
        if torch.any(low_stopband):
            low_s21_leak = torch.relu(pred_s21[:, low_stopband] - tgt_s21[:, low_stopband])
            low_leak_mean = torch.mean(low_s21_leak ** 2)
            low_leak_peak = torch.amax(low_s21_leak ** 2, dim=1).mean()
            low_s11_match = torch.relu(float(_LOW_FREQ_REFLECTION_LIMIT_DB) - pred_s11[:, low_stopband])
            low_s22_match = torch.relu(float(_LOW_FREQ_REFLECTION_LIMIT_DB) - pred_s22[:, low_stopband])
            low_match_loss = torch.mean(low_s11_match ** 2) + torch.mean(low_s22_match ** 2)
            loss = (
                loss
                + float(_TX_LOW_STOPBAND_MEAN_WEIGHT) * low_leak_mean
                + float(_TX_LOW_STOPBAND_PEAK_WEIGHT) * low_leak_peak
                + float(_LOW_FREQ_REFLECTION_WEIGHT) * low_match_loss
            )
        tv = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean() + torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        loss = loss + 0.05 * tv
        loss.backward()
        optimizer.step()
        if mask_tensor is not None:
            with torch.no_grad():
                logits.data.masked_fill_(mask_tensor.bool(), -12.0)

        current_layout = (torch.sigmoid(logits).detach().cpu().numpy()[0, 0] > 0.5).astype(np.float32)
        current_layout = _enforce_split_mask(current_layout, cfg)
        score = float(_resnet_algo.fitness_np(current_layout, surrogate, target_db, spec, args=args)[0])
        if score < best_score:
            best_score = score
            best_layout = current_layout.copy()
        if callback is not None and (step % 10 == 0 or step == steps - 1):
            soft_layout = prob.detach().cpu().numpy()[0, 0].copy()
            soft_layout = _enforce_split_mask(soft_layout, cfg)
            callback("gd", step, best_score, best_layout.copy(), soft_layout)

    return best_layout, best_score


# 변경: ResNet 참조 알고리즘 내부의 주요 훅을 분리선/제약 버전으로 monkey-patch
_resnet_algo.random_population      = _split_aware_random_population
_resnet_algo.build_seeded_population = _split_aware_build_seeded_population
_resnet_algo.mutate_population      = _split_aware_mutate_population
_resnet_algo.crossover_population   = _split_aware_crossover_population
_resnet_algo.run_bpso               = _split_aware_run_bpso
_resnet_algo.run_gd                 = _split_aware_run_gd
# 변경: 핵심! Custom fitness를 reference에 주입해 IndexError(axis mismatch) 해결
_resnet_algo.fitness_np             = _constraint_fitness_np
_resnet_algo.retrieval_seed         = _constraint_retrieval_seed
