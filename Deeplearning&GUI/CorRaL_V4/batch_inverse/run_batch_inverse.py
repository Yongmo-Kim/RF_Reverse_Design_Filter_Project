#!/usr/bin/env python3
"""
Headless inverse design runner for CorRaL V4 forward ensembles.

This script is intended for supercomputer batch jobs. It loads the trained
25x25 ResNet ensemble, searches many binary layouts without opening the GUI,
and saves the best candidates plus prediction plots for EM verification.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
GUI_DIR = SCRIPT_DIR.parent / "gui_resnet"
if str(GUI_DIR) not in sys.path:
    sys.path.insert(0, str(GUI_DIR))

from models import ResNet18_25x25  # noqa: E402


GRID_SIZE = 25
LAYOUT_PIXELS = GRID_SIZE * GRID_SIZE
DB_KEYS = ("s11_db", "s12_db", "s21_db", "s22_db")
PHASE_KEYS = ("s11_deg", "s12_deg", "s21_deg", "s22_deg")


@dataclass
class TargetSpec:
    f_low: float
    f_high: float
    insertion_loss: float
    return_loss: float
    stopband_loss: float
    transition_ratio: float

    @property
    def stopband_level(self) -> float:
        # GUI convention: IL - 3 dB plus user stopband loss.
        return float(self.insertion_loss) - 3.0 + float(self.stopband_loss)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CorRaL V4 headless inverse design")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--f-low", type=float, required=True)
    parser.add_argument("--f-high", type=float, required=True)
    parser.add_argument("--insertion-loss", type=float, default=-3.0)
    parser.add_argument("--return-loss", type=float, default=-10.0)
    parser.add_argument("--stopband-loss", type=float, default=-12.0)
    parser.add_argument("--transition-ratio", type=float, default=0.10)
    parser.add_argument(
        "--algorithm",
        default="ga_dbs",
        help="Search algorithm. Available: ga_gd, bpso_gd, ga_dbs.",
    )
    parser.add_argument("--pop-size", type=int, default=512)
    parser.add_argument("--generations", type=int, default=160)
    parser.add_argument("--mutation-rate", type=float, default=0.025)
    parser.add_argument("--elite-frac", type=float, default=0.10)
    parser.add_argument("--dbs-top", type=int, default=32)
    parser.add_argument("--dbs-passes", type=int, default=2)
    parser.add_argument("--gd-top", type=int, default=32)
    parser.add_argument("--gd-steps", type=int, default=240)
    parser.add_argument("--gd-lr", type=float, default=0.08)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--min-hamming-frac",
        type=float,
        default=0.08,
        help="Minimum pixel difference fraction between saved top candidates. 0.08 means about 50 pixels.",
    )
    parser.add_argument(
        "--min-response-rms",
        type=float,
        default=1.25,
        help="Minimum S-parameter dB RMS difference between saved top candidates.",
    )
    parser.add_argument(
        "--archive-extra-random",
        type=int,
        default=96,
        help="Also archive this many random candidates per generation so top-k can be diverse.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-final", action="store_true", help="Also load forward_final.pt if present")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_state_dict(obj) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            value = obj.get(key)
            if isinstance(value, dict):
                obj = value
                break
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(obj)!r}")

    state = {}
    for key, value in obj.items():
        if not torch.is_tensor(value):
            continue
        clean_key = key
        for prefix in ("module.", "model.", "net."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        state[clean_key] = value
    return state


class ForwardEnsemble:
    def __init__(self, model_dir: Path, device: torch.device, use_final: bool = False):
        self.model_dir = model_dir
        self.device = device
        self.config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.freqs = np.load(model_dir / self.config.get("frequencies_file", "frequencies.npy")).astype(np.float32)
        self.target_keys = list(self.config["target_keys"])
        self.n_freq = int(self.config["freq_points"])
        self.output_dim = int(self.config["output_dim"])
        self.norm = self.config["normalization"]

        checkpoint_paths = [model_dir / rel for rel in self.config.get("ensemble_checkpoints", [])]
        if use_final and (model_dir / "forward_final.pt").exists():
            checkpoint_paths.append(model_dir / "forward_final.pt")
        if not checkpoint_paths:
            checkpoint_paths = sorted((model_dir / "ensemble").glob("forward_fold*.pt"))
        if not checkpoint_paths:
            raise FileNotFoundError(f"No ensemble checkpoint found under {model_dir}")

        self.models = []
        for ckpt in checkpoint_paths:
            model = ResNet18_25x25(output_dim=self.output_dim, pretrained=False)
            try:
                raw = torch.load(ckpt, map_location="cpu", weights_only=False)
            except TypeError:
                raw = torch.load(ckpt, map_location="cpu")
            state = normalize_state_dict(raw)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"[WARN] {ckpt.name}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
            model.to(device).eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self.models.append(model)
            print(f"Loaded {ckpt}", flush=True)

    def denormalize(self, y: np.ndarray) -> np.ndarray:
        y = y.reshape(y.shape[0], len(self.target_keys), self.n_freq).astype(np.float32)
        out = np.empty_like(y)
        for i, key in enumerate(self.target_keys):
            lo = float(self.norm[key]["min"])
            hi = float(self.norm[key]["max"])
            out[:, i, :] = y[:, i, :] * (hi - lo) + lo
            if key.endswith("_db"):
                out[:, i, :] = np.minimum(out[:, i, :], 0.0)
        return out

    def scale_std(self, y: np.ndarray) -> np.ndarray:
        y = y.reshape(y.shape[0], len(self.target_keys), self.n_freq).astype(np.float32)
        out = np.empty_like(y)
        for i, key in enumerate(self.target_keys):
            lo = float(self.norm[key]["min"])
            hi = float(self.norm[key]["max"])
            out[:, i, :] = y[:, i, :] * abs(hi - lo)
        return out

    @torch.no_grad()
    def predict(self, layouts: np.ndarray, chunk_size: int) -> Tuple[np.ndarray, np.ndarray]:
        layouts = layouts.astype(np.float32, copy=False)
        means: List[np.ndarray] = []
        stds: List[np.ndarray] = []
        for start in range(0, len(layouts), chunk_size):
            batch = layouts[start:start + chunk_size]
            x = torch.from_numpy(batch[:, None, :, :]).to(self.device)
            preds = []
            for model in self.models:
                preds.append(model(x).detach().float().cpu().numpy())
            stack = np.stack(preds, axis=0)
            means.append(self.denormalize(stack.mean(axis=0)))
            stds.append(self.scale_std(stack.std(axis=0)))
        return np.concatenate(means, axis=0), np.concatenate(stds, axis=0)

    def channel_index(self, key: str) -> int:
        return self.target_keys.index(key)


def band_masks(freqs: np.ndarray, spec: TargetSpec) -> Dict[str, np.ndarray]:
    pb = (freqs >= spec.f_low) & (freqs <= spec.f_high)
    tr_low = spec.f_low * (1.0 - spec.transition_ratio)
    tr_high = spec.f_high * (1.0 + spec.transition_ratio)
    trans = ((freqs >= tr_low) & (freqs < spec.f_low)) | ((freqs > spec.f_high) & (freqs <= tr_high))
    sb_left = freqs < tr_low
    sb_right = freqs > tr_high
    sb = sb_left | sb_right
    return {"pb": pb, "trans": trans, "sb": sb, "sb_left": sb_left, "sb_right": sb_right}


def mean_positive(x: np.ndarray, axis=None) -> np.ndarray:
    return np.maximum(x, 0.0).mean(axis=axis)


def topk_positive_mean(x: np.ndarray, k: int = 3) -> np.ndarray:
    x = np.maximum(x, 0.0)
    if x.shape[1] == 0:
        return np.zeros(x.shape[0], dtype=np.float32)
    k = min(k, x.shape[1])
    part = np.partition(x, -k, axis=1)[:, -k:]
    return part.mean(axis=1)


def layout_regularization(layouts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    density = layouts.mean(axis=(1, 2))
    density_penalty = np.abs(density - 0.50)
    tv = np.abs(np.diff(layouts, axis=1)).mean(axis=(1, 2)) + np.abs(np.diff(layouts, axis=2)).mean(axis=(1, 2))
    return density_penalty, tv


def score_batch(
    pred: np.ndarray,
    layouts: np.ndarray,
    ensemble_std: np.ndarray,
    model: ForwardEnsemble,
    spec: TargetSpec,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    freqs = model.freqs
    masks = band_masks(freqs, spec)
    s11 = pred[:, model.channel_index("s11_db"), :]
    s12 = pred[:, model.channel_index("s12_db"), :]
    s21 = pred[:, model.channel_index("s21_db"), :]
    s22 = pred[:, model.channel_index("s22_db"), :]

    pb = masks["pb"]
    sb = masks["sb"]
    sb_left = masks["sb_left"]
    sb_right = masks["sb_right"]
    zeros = np.zeros(len(layouts), dtype=np.float32)

    # Transmission: passband must be near insertion_loss, stopband must be below stopband_level.
    s21_pb_under = mean_positive(spec.insertion_loss - s21[:, pb], axis=1)
    s12_pb_under = mean_positive(spec.insertion_loss - s12[:, pb], axis=1)
    s21_pb_abs = np.abs(s21[:, pb] - spec.insertion_loss).mean(axis=1)
    s12_pb_abs = np.abs(s12[:, pb] - spec.insertion_loss).mean(axis=1)

    s21_sb = mean_positive(s21[:, sb] - spec.stopband_level, axis=1)
    s12_sb = mean_positive(s12[:, sb] - spec.stopband_level, axis=1)
    s21_left = mean_positive(s21[:, sb_left] - spec.stopband_level, axis=1) if sb_left.any() else zeros
    s12_left = mean_positive(s12[:, sb_left] - spec.stopband_level, axis=1) if sb_left.any() else zeros
    s21_right = mean_positive(s21[:, sb_right] - spec.stopband_level, axis=1) if sb_right.any() else zeros
    s12_right = mean_positive(s12[:, sb_right] - spec.stopband_level, axis=1) if sb_right.any() else zeros
    s21_sb_worst = topk_positive_mean(s21[:, sb] - spec.stopband_level)
    s12_sb_worst = topk_positive_mean(s12[:, sb] - spec.stopband_level)

    # Reflection: passband return loss must be below return_loss.
    s11_refl = mean_positive(s11[:, pb] - spec.return_loss, axis=1)
    s22_refl = mean_positive(s22[:, pb] - spec.return_loss, axis=1)
    s11_worst = topk_positive_mean(s11[:, pb] - spec.return_loss)
    s22_worst = topk_positive_mean(s22[:, pb] - spec.return_loss)

    reciprocity = np.abs(s21 - s12).mean(axis=1)
    ripple = s21[:, pb].std(axis=1) + s12[:, pb].std(axis=1)
    uncertainty = ensemble_std[:, [model.channel_index(k) for k in DB_KEYS], :].mean(axis=(1, 2))
    density_penalty, tv_penalty = layout_regularization(layouts)

    parts: Dict[str, np.ndarray] = {
        "s21_pb_under": np.asarray(s21_pb_under),
        "s12_pb_under": np.asarray(s12_pb_under),
        "s21_pb_abs": np.asarray(s21_pb_abs),
        "s12_pb_abs": np.asarray(s12_pb_abs),
        "s21_sb": np.asarray(s21_sb),
        "s12_sb": np.asarray(s12_sb),
        "s21_left_sb": np.asarray(s21_left),
        "s12_left_sb": np.asarray(s12_left),
        "s21_right_sb": np.asarray(s21_right),
        "s12_right_sb": np.asarray(s12_right),
        "s21_sb_worst": np.asarray(s21_sb_worst),
        "s12_sb_worst": np.asarray(s12_sb_worst),
        "s11_refl": np.asarray(s11_refl),
        "s22_refl": np.asarray(s22_refl),
        "s11_worst": np.asarray(s11_worst),
        "s22_worst": np.asarray(s22_worst),
        "reciprocity": np.asarray(reciprocity),
        "ripple": np.asarray(ripple),
        "uncertainty": np.asarray(uncertainty),
        "density": np.asarray(density_penalty),
        "tv": np.asarray(tv_penalty),
    }

    # The score balances transmission and matching. Earlier versions were too
    # transmission-heavy and accepted layouts with poor S11/S22 in the passband.
    score = (
        260.0 * (parts["s21_pb_under"] + parts["s12_pb_under"])
        + 40.0 * (parts["s21_pb_abs"] + parts["s12_pb_abs"])
        + 120.0 * (parts["s21_sb"] + parts["s12_sb"])
        + 180.0 * (parts["s21_left_sb"] + parts["s12_left_sb"])
        + 100.0 * (parts["s21_right_sb"] + parts["s12_right_sb"])
        + 80.0 * (parts["s21_sb_worst"] + parts["s12_sb_worst"])
        + 260.0 * (parts["s11_refl"] + parts["s22_refl"])
        + 130.0 * (parts["s11_worst"] + parts["s22_worst"])
        + 20.0 * parts["reciprocity"]
        + 10.0 * parts["ripple"]
        + 15.0 * parts["uncertainty"]
        + 20.0 * parts["density"]
        + 0.5 * parts["tv"]
    )
    return score.astype(np.float32), parts


def evaluate_layouts(
    layouts: np.ndarray,
    ensemble: ForwardEnsemble,
    spec: TargetSpec,
    chunk_size: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    pred, std = ensemble.predict(layouts, chunk_size)
    score, parts = score_batch(pred, layouts, std, ensemble, spec)
    return score, parts, pred


def make_population(pop_size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=(pop_size, GRID_SIZE, GRID_SIZE), dtype=np.uint8)


def breed_population(
    population: np.ndarray,
    scores: np.ndarray,
    pop_size: int,
    mutation_rate: float,
    elite_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    elite_count = max(4, int(math.ceil(pop_size * elite_frac)))
    elite_idx = np.argsort(scores)[:elite_count]
    elites = population[elite_idx]
    children = [elites]

    inv = 1.0 / (scores[elite_idx] - scores[elite_idx].min() + 1.0)
    parent_prob = inv / inv.sum()
    while sum(len(c) for c in children) < pop_size:
        n = min(pop_size - sum(len(c) for c in children), elite_count * 2)
        a = elites[rng.choice(elite_count, size=n, p=parent_prob)]
        b = elites[rng.choice(elite_count, size=n, p=parent_prob)]
        mask = rng.random(size=a.shape) < 0.5
        child = np.where(mask, a, b).astype(np.uint8)
        mut = rng.random(size=child.shape) < mutation_rate
        child[mut] = 1 - child[mut]
        children.append(child)
    return np.concatenate(children, axis=0)[:pop_size]


def collect_records(
    records: List[Tuple[float, np.ndarray, Dict[str, float], np.ndarray]],
    layouts: np.ndarray,
    scores: np.ndarray,
    parts: Dict[str, np.ndarray],
    preds: np.ndarray,
    rng: np.random.Generator | None = None,
    extra_random: int = 0,
) -> None:
    order = np.argsort(scores)
    keep_list = list(order[: min(len(order), 64)])
    if rng is not None and extra_random > 0 and len(order) > len(keep_list):
        pool = order[len(keep_list):]
        n_extra = min(extra_random, len(pool))
        keep_list.extend(rng.choice(pool, size=n_extra, replace=False).tolist())
    for idx in keep_list:
        comp = {key: float(value[idx]) for key, value in parts.items()}
        records.append((float(scores[idx]), layouts[idx].copy(), comp, preds[idx].copy()))


def ga_search(
    ensemble: ForwardEnsemble,
    spec: TargetSpec,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> List[Tuple[float, np.ndarray, Dict[str, float], np.ndarray]]:
    population = make_population(args.pop_size, rng)
    records: List[Tuple[float, np.ndarray, Dict[str, float], np.ndarray]] = []
    best_score = float("inf")

    for gen in range(args.generations + 1):
        scores, parts, preds = evaluate_layouts(population, ensemble, spec, args.chunk_size)
        collect_records(records, population, scores, parts, preds, rng, args.archive_extra_random)
        cur_best = float(scores.min())
        if cur_best < best_score:
            best_score = cur_best
        if gen % 10 == 0 or gen == args.generations:
            print(f"GA gen={gen:04d} best={cur_best:.4f} global_best={best_score:.4f}", flush=True)
        if gen == args.generations:
            break
        anneal = 1.0 - (gen / max(args.generations, 1))
        mutation = max(args.mutation_rate * (0.25 + 0.75 * anneal), 0.004)
        population = breed_population(population, scores, args.pop_size, mutation, args.elite_frac, rng)
    return records


def bpso_search(
    ensemble: ForwardEnsemble,
    spec: TargetSpec,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> List[Tuple[float, np.ndarray, Dict[str, float], np.ndarray]]:
    n = args.pop_size
    positions = rng.random(size=(n, GRID_SIZE, GRID_SIZE)).astype(np.float32)
    velocity = rng.normal(0.0, 0.5, size=positions.shape).astype(np.float32)
    pbest_layout = (positions > 0.5).astype(np.uint8)
    pbest_score = np.full(n, np.inf, dtype=np.float32)
    gbest_layout = pbest_layout[0].copy()
    gbest_score = float("inf")
    records: List[Tuple[float, np.ndarray, Dict[str, float], np.ndarray]] = []

    for it in range(args.generations + 1):
        layouts = (positions > 0.5).astype(np.uint8)
        scores, parts, preds = evaluate_layouts(layouts, ensemble, spec, args.chunk_size)
        collect_records(records, layouts, scores, parts, preds, rng, args.archive_extra_random)

        improved = scores < pbest_score
        pbest_score[improved] = scores[improved]
        pbest_layout[improved] = layouts[improved]

        best_idx = int(np.argmin(scores))
        if float(scores[best_idx]) < gbest_score:
            gbest_score = float(scores[best_idx])
            gbest_layout = layouts[best_idx].copy()

        if it % 10 == 0 or it == args.generations:
            print(f"BPSO iter={it:04d} best={float(scores.min()):.4f} global_best={gbest_score:.4f}", flush=True)
        if it == args.generations:
            break

        progress = it / max(args.generations, 1)
        inertia = 0.9 - 0.5 * progress
        c1 = 1.6
        c2 = 1.8
        r1 = rng.random(size=positions.shape).astype(np.float32)
        r2 = rng.random(size=positions.shape).astype(np.float32)
        velocity = (
            inertia * velocity
            + c1 * r1 * (pbest_layout.astype(np.float32) - positions)
            + c2 * r2 * (gbest_layout.astype(np.float32)[None, :, :] - positions)
        )
        velocity = np.clip(velocity, -6.0, 6.0)
        prob = 1.0 / (1.0 + np.exp(-velocity))
        positions = (rng.random(size=positions.shape) < prob).astype(np.float32)

    return records


def random_search(
    ensemble: ForwardEnsemble,
    spec: TargetSpec,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> List[Tuple[float, np.ndarray, Dict[str, float], np.ndarray]]:
    records: List[Tuple[float, np.ndarray, Dict[str, float], np.ndarray]] = []
    best_score = float("inf")
    for batch_id in range(args.generations + 1):
        population = make_population(args.pop_size, rng)
        scores, parts, preds = evaluate_layouts(population, ensemble, spec, args.chunk_size)
        collect_records(records, population, scores, parts, preds, rng, args.archive_extra_random)
        cur_best = float(scores.min())
        if cur_best < best_score:
            best_score = cur_best
        if batch_id % 10 == 0 or batch_id == args.generations:
            tested = (batch_id + 1) * args.pop_size
            print(f"RANDOM batch={batch_id:04d} tested={tested} best={cur_best:.4f} global_best={best_score:.4f}", flush=True)
    return records


def torch_denorm_channel(y: torch.Tensor, ensemble: ForwardEnsemble, key: str) -> torch.Tensor:
    idx = ensemble.channel_index(key)
    lo = float(ensemble.norm[key]["min"])
    hi = float(ensemble.norm[key]["max"])
    out = y[:, idx, :] * (hi - lo) + lo
    if key.endswith("_db"):
        out = torch.minimum(out, torch.zeros_like(out))
    return out


def gd_loss_from_logits(logits: torch.Tensor, ensemble: ForwardEnsemble, spec: TargetSpec) -> torch.Tensor:
    x = torch.sigmoid(logits).view(1, 1, GRID_SIZE, GRID_SIZE)
    outs = []
    for model in ensemble.models:
        outs.append(model(x))
    y = torch.stack(outs, dim=0).mean(dim=0).view(1, len(ensemble.target_keys), ensemble.n_freq)

    device = logits.device
    freqs = torch.as_tensor(ensemble.freqs, dtype=torch.float32, device=device)
    pb = (freqs >= spec.f_low) & (freqs <= spec.f_high)
    tr_low = spec.f_low * (1.0 - spec.transition_ratio)
    tr_high = spec.f_high * (1.0 + spec.transition_ratio)
    sb_left = freqs < tr_low
    sb_right = freqs > tr_high
    sb = sb_left | sb_right

    s11 = torch_denorm_channel(y, ensemble, "s11_db")
    s12 = torch_denorm_channel(y, ensemble, "s12_db")
    s21 = torch_denorm_channel(y, ensemble, "s21_db")
    s22 = torch_denorm_channel(y, ensemble, "s22_db")

    z = torch.zeros((), dtype=torch.float32, device=device)
    tx_pb = torch.relu(spec.insertion_loss - s21[:, pb]).mean() + torch.relu(spec.insertion_loss - s12[:, pb]).mean()
    tx_pb_abs = (s21[:, pb] - spec.insertion_loss).abs().mean() + (s12[:, pb] - spec.insertion_loss).abs().mean()
    tx_sb = torch.relu(s21[:, sb] - spec.stopband_level).mean() + torch.relu(s12[:, sb] - spec.stopband_level).mean() if sb.any() else z
    tx_left = torch.relu(s21[:, sb_left] - spec.stopband_level).mean() + torch.relu(s12[:, sb_left] - spec.stopband_level).mean() if sb_left.any() else z
    tx_right = torch.relu(s21[:, sb_right] - spec.stopband_level).mean() + torch.relu(s12[:, sb_right] - spec.stopband_level).mean() if sb_right.any() else z
    refl = torch.relu(s11[:, pb] - spec.return_loss).mean() + torch.relu(s22[:, pb] - spec.return_loss).mean()
    recip = (s21 - s12).abs().mean()
    ripple = s21[:, pb].std(unbiased=False) + s12[:, pb].std(unbiased=False)
    binary = (x * (1.0 - x)).mean()
    density = (x.mean() - 0.5).abs()

    return (
        260.0 * tx_pb
        + 40.0 * tx_pb_abs
        + 120.0 * tx_sb
        + 180.0 * tx_left
        + 100.0 * tx_right
        + 80.0 * refl
        + 20.0 * recip
        + 10.0 * ripple
        + 5.0 * binary
        + 20.0 * density
    )


def gd_refine_one(
    layout: np.ndarray,
    ensemble: ForwardEnsemble,
    spec: TargetSpec,
    args: argparse.Namespace,
) -> Tuple[float, np.ndarray, Dict[str, float], np.ndarray]:
    device = ensemble.device
    init = np.where(layout.astype(np.float32) > 0.5, 2.0, -2.0)
    logits = torch.tensor(init, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=args.gd_lr)

    best_layout = layout.copy()
    best_score = float("inf")
    best_parts: Dict[str, float] = {}
    best_pred = None

    for step in range(args.gd_steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = gd_loss_from_logits(logits, ensemble, spec)
        loss.backward()
        opt.step()
        with torch.no_grad():
            logits.clamp_(-8.0, 8.0)

        if step % 20 == 0 or step == args.gd_steps:
            with torch.no_grad():
                hard = (torch.sigmoid(logits).detach().cpu().numpy() > 0.5).astype(np.uint8)
            scores, parts, preds = evaluate_layouts(hard[None, :, :], ensemble, spec, args.chunk_size)
            if float(scores[0]) < best_score:
                best_score = float(scores[0])
                best_layout = hard.copy()
                best_parts = {key: float(value[0]) for key, value in parts.items()}
                best_pred = preds[0].copy()
            print(f"GD step={step:04d} loss={float(loss.detach().cpu()):.4f} best_score={best_score:.4f}", flush=True)

    if best_pred is None:
        scores, parts, preds = evaluate_layouts(best_layout[None, :, :], ensemble, spec, args.chunk_size)
        best_score = float(scores[0])
        best_parts = {key: float(value[0]) for key, value in parts.items()}
        best_pred = preds[0].copy()
    return best_score, best_layout, best_parts, best_pred


def dbs_refine_one(
    layout: np.ndarray,
    ensemble: ForwardEnsemble,
    spec: TargetSpec,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[float, np.ndarray, Dict[str, float], np.ndarray]:
    current = layout.copy()
    score, parts, pred = evaluate_layouts(current[None, :, :], ensemble, spec, args.chunk_size)
    best_score = float(score[0])
    best_parts = {key: float(value[0]) for key, value in parts.items()}
    best_pred = pred[0].copy()

    for pass_id in range(args.dbs_passes):
        improved = 0
        order = rng.permutation(LAYOUT_PIXELS)
        for start in range(0, LAYOUT_PIXELS, args.chunk_size):
            pix = order[start:start + args.chunk_size]
            candidates = np.repeat(current[None, :, :], len(pix), axis=0)
            rows = pix // GRID_SIZE
            cols = pix % GRID_SIZE
            candidates[np.arange(len(pix)), rows, cols] = 1 - candidates[np.arange(len(pix)), rows, cols]
            scores, cand_parts, preds = evaluate_layouts(candidates, ensemble, spec, args.chunk_size)
            local_idx = int(np.argmin(scores))
            if float(scores[local_idx]) + 1e-6 < best_score:
                current = candidates[local_idx].copy()
                best_score = float(scores[local_idx])
                best_parts = {key: float(value[local_idx]) for key, value in cand_parts.items()}
                best_pred = preds[local_idx].copy()
                improved += 1
        print(f"DBS pass={pass_id + 1} score={best_score:.4f} accepted={improved}", flush=True)
        if improved == 0:
            break
    return best_score, current, best_parts, best_pred


def dedupe_records(
    records: List[Tuple[float, np.ndarray, Dict[str, float], np.ndarray]],
    top_k: int,
    min_hamming_frac: float = 0.0,
    min_response_rms: float = 0.0,
):
    seen = set()
    unique = []
    min_hamming = int(round(LAYOUT_PIXELS * max(min_hamming_frac, 0.0)))
    for score, layout, parts, pred in sorted(records, key=lambda x: x[0]):
        key = np.packbits(layout.reshape(-1)).tobytes()
        if key in seen:
            continue
        if min_hamming > 0:
            flat = layout.reshape(-1)
            too_close = False
            for _score, old_layout, _parts, _pred in unique:
                if int(np.count_nonzero(flat != old_layout.reshape(-1))) < min_hamming:
                    too_close = True
                    break
            if too_close:
                continue
        if min_response_rms > 0:
            db_pred = pred[:4].reshape(-1)
            too_similar_response = False
            for _score, _old_layout, _parts, old_pred in unique:
                rms = float(np.sqrt(np.mean((db_pred - old_pred[:4].reshape(-1)) ** 2)))
                if rms < min_response_rms:
                    too_similar_response = True
                    break
            if too_similar_response:
                continue
        seen.add(key)
        unique.append((score, layout, parts, pred))
        if len(unique) >= top_k:
            break
    if len(unique) < top_k and (min_hamming > 0 or min_response_rms > 0):
        relaxed = dedupe_records(records, top_k, min_hamming_frac=0.0, min_response_rms=0.0)
        seen_keys = {np.packbits(item[1].reshape(-1)).tobytes() for item in unique}
        for item in relaxed:
            key = np.packbits(item[1].reshape(-1)).tobytes()
            if key not in seen_keys:
                unique.append(item)
                seen_keys.add(key)
            if len(unique) >= top_k:
                break
    return unique


def prediction_metrics(pred: np.ndarray, ensemble: ForwardEnsemble, spec: TargetSpec) -> Dict[str, float]:
    masks = band_masks(ensemble.freqs, spec)
    get = lambda key: pred[ensemble.channel_index(key)]
    out = {}
    for key in DB_KEYS:
        arr = get(key)
        out[f"{key}_pb_mean"] = float(arr[masks["pb"]].mean())
        out[f"{key}_pb_min"] = float(arr[masks["pb"]].min())
        out[f"{key}_pb_max"] = float(arr[masks["pb"]].max())
        out[f"{key}_sb_left_mean"] = float(arr[masks["sb_left"]].mean()) if masks["sb_left"].any() else float("nan")
        out[f"{key}_sb_right_mean"] = float(arr[masks["sb_right"]].mean()) if masks["sb_right"].any() else float("nan")
    return out


def save_prediction_csv(path: Path, freqs: np.ndarray, pred: np.ndarray, target_keys: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["freq_ghz", *target_keys])
        for i, freq in enumerate(freqs):
            writer.writerow([float(freq), *[float(pred[j, i]) for j in range(len(target_keys))]])


def save_plot(path: Path, freqs: np.ndarray, pred: np.ndarray, ensemble: ForwardEnsemble, spec: TargetSpec, plt) -> None:
    masks = band_masks(freqs, spec)
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    plot_keys = ["s11_db", "s22_db", "s21_db", "s12_db"]
    titles = ["S11 Return Loss", "S22 Return Loss", "S21 Insertion Loss", "S12 Insertion Loss"]
    for ax, key, title in zip(axes, plot_keys, titles):
        y = pred[ensemble.channel_index(key)]
        ax.plot(freqs, y, label="AI pred", color="#1f77b4")
        ax.axvspan(spec.f_low, spec.f_high, color="#d9f2df", alpha=0.75)
        if key in ("s11_db", "s22_db"):
            ax.axhline(spec.return_loss, color="black", linestyle="--", label="target")
            ax.set_ylim(-30, 0)
        else:
            ax.axhline(spec.insertion_loss, color="black", linestyle="--", label="pass target")
            ax.axhline(spec.stopband_level, color="#991f1f", linestyle="--", label="stop target")
            ax.set_ylim(min(-30, spec.stopband_level - 5), 2)
            for mask in ("sb_left", "sb_right"):
                if masks[mask].any():
                    ax.axvspan(freqs[masks[mask]][0], freqs[masks[mask]][-1], color="#ffd8d8", alpha=0.35)
        ax.set_title(title)
        ax.set_ylabel("dB")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Frequency (GHz)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_outputs(
    records,
    ensemble: ForwardEnsemble,
    spec: TargetSpec,
    args: argparse.Namespace,
) -> None:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    top = dedupe_records(records, args.top_k, args.min_hamming_frac, args.min_response_rms)
    if not top:
        raise RuntimeError("No candidate records generated")

    plt = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        plt = _plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable, PNG plots will be skipped: {exc}", flush=True)

    summary_path = out / "summary.csv"
    metric_keys = sorted(top[0][2].keys())
    pred_metric_keys = sorted(prediction_metrics(top[0][3], ensemble, spec).keys())
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "score", *metric_keys, *pred_metric_keys, "layout_npy", "layout_csv", "prediction_csv", "plot_png"])
        for rank, (score, layout, parts, pred) in enumerate(top, start=1):
            stem = f"best_{rank:03d}"
            layout_npy = out / f"{stem}_layout.npy"
            layout_csv = out / f"{stem}_layout.csv"
            pred_csv = out / f"{stem}_prediction.csv"
            plot_png = out / f"{stem}.png"
            np.save(layout_npy, layout.astype(np.uint8))
            np.savetxt(layout_csv, layout.astype(int), fmt="%d", delimiter=",")
            save_prediction_csv(pred_csv, ensemble.freqs, pred, ensemble.target_keys)
            if plt is not None:
                save_plot(plot_png, ensemble.freqs, pred, ensemble, spec, plt)
            pred_metrics = prediction_metrics(pred, ensemble, spec)
            writer.writerow(
                [rank, score]
                + [parts[k] for k in metric_keys]
                + [pred_metrics[k] for k in pred_metric_keys]
                + [str(layout_npy), str(layout_csv), str(pred_csv), str(plot_png)]
            )

    np.savez_compressed(
        out / "best_candidates.npz",
        layouts=np.stack([x[1] for x in top]).astype(np.uint8),
        predictions=np.stack([x[3] for x in top]).astype(np.float32),
        scores=np.array([x[0] for x in top], dtype=np.float32),
        frequencies=ensemble.freqs,
        target_keys=np.array(ensemble.target_keys),
        target_json=json.dumps(spec.__dict__),
    )
    (out / "run_config.json").write_text(
        json.dumps(
            {
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "target": spec.__dict__,
                "model_dir": str(ensemble.model_dir),
                "n_models": len(ensemble.models),
                "freq_min": float(ensemble.freqs.min()),
                "freq_max": float(ensemble.freqs.max()),
                "freq_points": int(len(ensemble.freqs)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved summary: {summary_path}", flush=True)
    print(f"Saved best bundle: {out / 'best_candidates.npz'}", flush=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    spec = TargetSpec(
        f_low=args.f_low,
        f_high=args.f_high,
        insertion_loss=args.insertion_loss,
        return_loss=args.return_loss,
        stopband_loss=args.stopband_loss,
        transition_ratio=args.transition_ratio,
    )

    print("========================================", flush=True)
    print("CorRaL V4 headless inverse design", flush=True)
    print(f"model_dir={args.model_dir}", flush=True)
    print(f"output_dir={args.output_dir}", flush=True)
    print(f"device={device}", flush=True)
    print(f"target={spec}", flush=True)
    print("========================================", flush=True)

    start = time.time()
    ensemble = ForwardEnsemble(args.model_dir, device, use_final=args.use_final)
    rng = np.random.default_rng(args.seed)

    args.algorithm = args.algorithm.lower()

    if args.algorithm in ("ga_gd", "ga_dbs"):
        records = ga_search(ensemble, spec, args, rng)
    elif args.algorithm == "bpso_gd":
        records = bpso_search(ensemble, spec, args, rng)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")

    if args.algorithm.endswith("_gd"):
        seeds = dedupe_records(records, args.gd_top, min_hamming_frac=0.0)
        print(f"GD refine seeds={len(seeds)}", flush=True)
        refined = []
        for i, (score, layout, _parts, _pred) in enumerate(seeds, start=1):
            print(f"GD seed {i}/{len(seeds)} start_score={score:.4f}", flush=True)
            refined.append(gd_refine_one(layout, ensemble, spec, args))
        records.extend(refined)
    elif args.algorithm.endswith("_dbs"):
        seeds = dedupe_records(records, args.dbs_top, min_hamming_frac=0.0)
        print(f"DBS refine seeds={len(seeds)}", flush=True)
        refined = []
        for i, (score, layout, _parts, _pred) in enumerate(seeds, start=1):
            print(f"DBS seed {i}/{len(seeds)} start_score={score:.4f}", flush=True)
            refined.append(dbs_refine_one(layout, ensemble, spec, args, rng))
        records.extend(refined)
    else:
        print("Refinement skipped by algorithm setting", flush=True)

    save_outputs(records, ensemble, spec, args)
    print(f"Done. elapsed_sec={time.time() - start:.1f}", flush=True)


if __name__ == "__main__":
    main()
