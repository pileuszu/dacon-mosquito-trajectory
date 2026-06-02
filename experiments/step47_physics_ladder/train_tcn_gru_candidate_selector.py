from __future__ import annotations

import argparse
import csv
import hashlib
import json
import copy
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


R_HIT = 0.01
EPS = 1e-8
SEQ_FEATURE_NAMES = [
    "speed",
    "prev_speed_ratio",
    "acc_norm_over_speed",
    "acc_parallel_over_speed",
    "acc_perp_over_speed",
    "jerk_over_speed",
    "turn_cos",
    "curvature",
    "direction_flag",
]
CAND_FEATURE_NAMES = [
    "candidate_parallel_over_speed_horizon",
    "candidate_perp_over_speed_horizon",
    "candidate_dist_over_speed_horizon",
    "candidate_d1_coeff",
    "candidate_parallel_coeff",
    "candidate_perp_coeff",
    "candidate_d2_coeff",
    "candidate_jerk_coeff",
    "candidate_time_scale",
    "ctx_speed",
    "ctx_prev_speed_ratio",
    "ctx_acc_norm_over_speed",
    "ctx_acc_parallel_over_speed",
    "ctx_acc_perp_over_speed",
    "ctx_jerk_over_speed",
    "ctx_turn_cos",
    "ctx_curvature",
    "ctx_direction_flag",
    "coeff_parallel_x_ctx_acc_parallel",
    "coeff_perp_x_ctx_acc_perp",
]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    d1: float
    par: float
    perp: float
    d2: float = 0.0
    jerk: float = 0.0
    time_scale: float = 1.0


CANDIDATES = [
    CandidateSpec("p0_2d1", 2.00, 0.00, 0.00),
    CandidateSpec("acc_2d1_040", 2.00, 0.40, 0.40),
    CandidateSpec("acc_2d1_050", 2.00, 0.50, 0.50),
    CandidateSpec("acc_2d1_056", 1.98, 0.56, 0.56),
    CandidateSpec("acc_2d1_060", 2.00, 0.60, 0.60),
    CandidateSpec("frenet_best", 1.98, 0.96, -0.08),
    CandidateSpec("frenet_par090_perp000", 1.98, 0.90, 0.00),
    CandidateSpec("frenet_par100_perp000", 1.98, 1.00, 0.00),
    CandidateSpec("frenet_par100_perp_neg010", 2.00, 1.00, -0.10),
    CandidateSpec("frenet_par090_perp020", 1.96, 0.90, 0.20),
    CandidateSpec("frenet_par080_perp020", 2.02, 0.80, 0.20),
    CandidateSpec("frenet_par110_perp_neg020", 1.94, 1.10, -0.20),
    CandidateSpec("frenet_fast_par100", 2.06, 1.00, -0.08),
    CandidateSpec("frenet_slow_par100", 1.90, 1.00, -0.08),
    CandidateSpec("jerk_small_pos", 1.98, 0.80, -0.05, jerk=0.08),
    CandidateSpec("jerk_small_neg", 1.98, 0.80, -0.05, jerk=-0.08),
    CandidateSpec("frenet_par070_perp_neg020", 1.98, 0.70, -0.20),
    CandidateSpec("frenet_par120_perp_neg020", 1.98, 1.20, -0.20),
    CandidateSpec("frenet_par120_perp020", 1.98, 1.20, 0.20),
    CandidateSpec("frenet_fast_par120_perp_neg020", 2.08, 1.20, -0.20),
    CandidateSpec("frenet_slow_par070_perp020", 1.86, 0.70, 0.20),
    CandidateSpec("latency_short_frenet_best_085", 1.98, 0.96, -0.08, time_scale=0.85),
    CandidateSpec("latency_short_frenet_best_092", 1.98, 0.96, -0.08, time_scale=0.92),
    CandidateSpec("latency_long_frenet_best_108", 1.98, 0.96, -0.08, time_scale=1.08),
    CandidateSpec("latency_long_frenet_best_115", 1.98, 0.96, -0.08, time_scale=1.15),
    CandidateSpec("latency_long_turn_neg_110", 1.98, 1.10, -0.20, time_scale=1.10),
    CandidateSpec("latency_short_turn_pos_090", 1.96, 0.90, 0.20, time_scale=0.90),
]

FAMILY_NAMES = ["base", "acc", "frenet", "turn", "jerk", "latency"]


def candidate_family_id(name: str) -> int:
    if name == "p0_2d1":
        return FAMILY_NAMES.index("base")
    if name.startswith("acc_"):
        return FAMILY_NAMES.index("acc")
    if name.startswith("latency"):
        return FAMILY_NAMES.index("latency")
    if "jerk" in name:
        return FAMILY_NAMES.index("jerk")
    if name.startswith("frenet_par") or name.startswith("frenet_fast") or name.startswith("frenet_slow"):
        return FAMILY_NAMES.index("turn")
    return FAMILY_NAMES.index("frenet")


CANDIDATE_FAMILY = np.asarray([candidate_family_id(spec.name) for spec in CANDIDATES], dtype=np.int64)


def read_xyz_csv(path: Path) -> np.ndarray:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = sorted(reader, key=lambda r: float(r["timestep_ms"]))
        return np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=np.float32)


def read_labels(path: Path) -> tuple[list[str], np.ndarray]:
    ids: list[str] = []
    xyz: list[list[float]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["id"])
            xyz.append([float(row["x"]), float(row["y"]), float(row["z"])])
    return ids, np.asarray(xyz, dtype=np.float32)


def read_submission_ids(path: Path) -> list[str]:
    with path.open("r", newline="") as f:
        return [row["id"] for row in csv.DictReader(f)]


def metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float | int]:
    err = np.linalg.norm(pred - true, axis=1)
    return {
        "hit": float(np.mean(err <= R_HIT)),
        "hits": int(np.sum(err <= R_HIT)),
        "mean": float(np.mean(err)),
        "q50": float(np.quantile(err, 0.50)),
        "q75": float(np.quantile(err, 0.75)),
        "q90": float(np.quantile(err, 0.90)),
        "q95": float(np.quantile(err, 0.95)),
        "q99": float(np.quantile(err, 0.99)),
    }


def motion_terms(x: np.ndarray, end_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p0 = x[:, end_idx]
    d1 = x[:, end_idx] - x[:, end_idx - 1]
    d2 = x[:, end_idx - 1] - x[:, end_idx - 2]
    acc = d1 - d2
    return p0, d1, acc


def make_candidates(x: np.ndarray, end_idx: int, horizon: int = 2) -> np.ndarray:
    p0, d1, acc = motion_terms(x, end_idx)
    d2 = x[:, end_idx - 1] - x[:, end_idx - 2]
    prev_acc = d2 - (x[:, end_idx - 2] - x[:, end_idx - 3])
    jerk = acc - prev_acc
    tangent = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + EPS)
    acc_par = np.sum(acc * tangent, axis=1, keepdims=True) * tangent
    acc_perp = acc - acc_par
    v_scale = horizon / 2.0
    acc_scale = (horizon / 2.0) ** 2
    preds = []
    for spec in CANDIDATES:
        spec_v_scale = v_scale * spec.time_scale
        spec_acc_scale = acc_scale * (spec.time_scale ** 2)
        preds.append(
            p0
            + spec.d1 * spec_v_scale * d1
            + spec.d2 * spec_v_scale * d2
            + spec.par * spec_acc_scale * acc_par
            + spec.perp * spec_acc_scale * acc_perp
            + spec.jerk * spec_acc_scale * jerk
        )
    return np.stack(preds, axis=1).astype(np.float32)


def stable_fold_id(sample_id: str, folds: int) -> int:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def candidate_spec_features(count: int) -> np.ndarray:
    rows = [[spec.d1, spec.par, spec.perp, spec.d2, spec.jerk, spec.time_scale] for spec in CANDIDATES]
    spec = np.asarray(rows, dtype=np.float32)[None, :, :]
    return np.repeat(spec, count, axis=0)


CANDIDATE_SPEC_MATRIX = candidate_spec_features(1)[0]


def turn_context_features(x: np.ndarray, end_idx: int) -> np.ndarray:
    _, d1, acc = motion_terms(x, end_idx)
    d2 = x[:, end_idx - 1] - x[:, end_idx - 2]
    prev_acc = d2 - (x[:, end_idx - 2] - x[:, end_idx - 3])
    tangent = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + EPS)
    acc_par_scalar = np.sum(acc * tangent, axis=1, keepdims=True)
    acc_par = acc_par_scalar * tangent
    acc_perp = acc - acc_par
    speed = np.linalg.norm(d1, axis=1, keepdims=True)
    prev_speed = np.linalg.norm(d2, axis=1, keepdims=True)
    acc_norm = np.linalg.norm(acc, axis=1, keepdims=True)
    perp_norm = np.linalg.norm(acc_perp, axis=1, keepdims=True)
    jerk_norm = np.linalg.norm(acc - prev_acc, axis=1, keepdims=True)
    turn_cos = np.sum(d1 * d2, axis=1, keepdims=True) / ((speed * prev_speed) + EPS)
    curvature = perp_norm / (speed + EPS)
    return np.concatenate(
        [speed, prev_speed, acc_norm, acc_par_scalar, perp_norm, jerk_norm, turn_cos, curvature],
        axis=1,
    ).astype(np.float32)


def turn_model_features_from_context(ctx: np.ndarray) -> np.ndarray:
    speed = ctx[:, 0:1]
    return np.concatenate(
        [
            speed,
            ctx[:, 1:2] / (speed + EPS),
            ctx[:, 2:3] / (speed + EPS),
            ctx[:, 3:4] / (speed + EPS),
            ctx[:, 4:5] / (speed + EPS),
            ctx[:, 5:6] / (speed + EPS),
            ctx[:, 6:7],
            ctx[:, 7:8],
        ],
        axis=1,
    ).astype(np.float32)


def recent_temporal_physics_features(x: np.ndarray, end_idx: int) -> np.ndarray:
    start = max(0, end_idx - 5)
    pts = x[:, start : end_idx + 1]
    v = np.diff(pts, axis=1)
    if v.shape[1] < 2:
        v = np.concatenate([v, v], axis=1)
    speeds = np.linalg.norm(v, axis=2)
    current_speed = speeds[:, -1:] + EPS
    path = np.sum(speeds, axis=1, keepdims=True)
    disp = np.linalg.norm(pts[:, -1] - pts[:, 0], axis=1, keepdims=True)
    straightness = disp / (path + EPS)
    speed_slope = (speeds[:, -1:] - speeds[:, :1]) / (np.mean(speeds, axis=1, keepdims=True) + EPS)
    speed_cv = np.std(speeds, axis=1, keepdims=True) / (np.mean(speeds, axis=1, keepdims=True) + EPS)
    v0 = v[:, :-1]
    v1 = v[:, 1:]
    turn_cos = np.sum(v0 * v1, axis=2) / ((np.linalg.norm(v0, axis=2) * np.linalg.norm(v1, axis=2)) + EPS)
    turn_accum = np.mean(1.0 - np.clip(turn_cos, -1.0, 1.0), axis=1, keepdims=True)
    acc = np.diff(v, axis=1)
    if acc.shape[1] == 0:
        accel_slope = np.zeros_like(current_speed)
    else:
        acc_norm = np.linalg.norm(acc, axis=2)
        accel_slope = (acc_norm[:, -1:] - acc_norm[:, :1]) / (np.mean(acc_norm, axis=1, keepdims=True) + EPS)
    return np.concatenate(
        [
            path / current_speed,
            straightness,
            speed_slope,
            speed_cv,
            turn_accum,
            accel_slope,
        ],
        axis=1,
    ).astype(np.float32)


def observation_environment_features(x: np.ndarray, end_idx: int) -> np.ndarray:
    start = max(0, end_idx - 5)
    pts = x[:, start : end_idx + 1]
    v = np.diff(pts, axis=1)
    if v.shape[1] < 3:
        pad = np.repeat(v[:, :1], 3 - v.shape[1], axis=1)
        v = np.concatenate([pad, v], axis=1)
    speed = np.linalg.norm(v, axis=2)
    mean_speed = np.mean(speed, axis=1, keepdims=True) + EPS
    path = np.sum(speed, axis=1, keepdims=True)
    disp = np.linalg.norm(pts[:, -1] - pts[:, 0], axis=1, keepdims=True)
    straightness = disp / (path + EPS)
    speed_cv = np.std(speed, axis=1, keepdims=True) / mean_speed
    v0 = v[:, :-1]
    v1 = v[:, 1:]
    turn_cos = np.sum(v0 * v1, axis=2) / ((np.linalg.norm(v0, axis=2) * np.linalg.norm(v1, axis=2)) + EPS)
    turn_volatility = np.std(1.0 - np.clip(turn_cos, -1.0, 1.0), axis=1, keepdims=True)
    linear_pred = x[:, end_idx - 1] + (x[:, end_idx - 1] - x[:, end_idx - 2])
    linear_resid = np.linalg.norm(x[:, end_idx] - linear_pred, axis=1, keepdims=True) / mean_speed
    acc = np.diff(v, axis=1)
    if acc.shape[1] < 2:
        jerk_vol = np.zeros_like(mean_speed)
    else:
        jerk = np.diff(acc, axis=1)
        jerk_vol = np.std(np.linalg.norm(jerk, axis=2), axis=1, keepdims=True) / mean_speed
    return np.concatenate([straightness, speed_cv, turn_volatility, linear_resid, jerk_vol], axis=1).astype(np.float32)


def fit_regime_bins(x: np.ndarray, end_idx: int) -> dict[str, list[float]]:
    ctx = turn_context_features(x, end_idx)
    temporal = recent_temporal_physics_features(x, end_idx)
    return {
        "speed": np.quantile(ctx[:, 0], [0.33, 0.66]).astype(float).tolist(),
        "curvature": np.quantile(ctx[:, 7], [0.40, 0.75]).astype(float).tolist(),
        "speed_slope": [float(np.quantile(temporal[:, 2], 0.50))],
    }


def assign_regimes(x: np.ndarray, end_idx: int, bins: dict[str, list[float]]) -> np.ndarray:
    ctx = turn_context_features(x, end_idx)
    temporal = recent_temporal_physics_features(x, end_idx)
    speed_bin = np.digitize(ctx[:, 0], np.asarray(bins["speed"], dtype=np.float32))
    curve_bin = np.digitize(ctx[:, 7], np.asarray(bins["curvature"], dtype=np.float32))
    fatigue_bin = np.digitize(temporal[:, 2], np.asarray(bins["speed_slope"], dtype=np.float32))
    return (speed_bin * 6 + curve_bin * 2 + fatigue_bin).astype(np.int64)


def candidate_regime_bias(
    candidates: np.ndarray,
    target: np.ndarray,
    regimes: np.ndarray,
    regime_count: int,
    shrink: float = 18.0,
) -> np.ndarray:
    err = np.linalg.norm(candidates - target[:, None, :], axis=2)
    global_hit = np.mean(err <= R_HIT, axis=0)
    global_mean = np.mean(err, axis=0)
    global_bias = np.log(global_hit + 1e-4) - 18.0 * global_mean
    out = np.zeros((regime_count, candidates.shape[1]), dtype=np.float32)
    for regime in range(regime_count):
        mask = regimes == regime
        if not np.any(mask):
            out[regime] = global_bias
            continue
        local_hit = np.mean(err[mask] <= R_HIT, axis=0)
        local_mean = np.mean(err[mask], axis=0)
        local_bias = np.log(local_hit + 1e-4) - 18.0 * local_mean
        alpha = float(np.sum(mask) / (np.sum(mask) + shrink))
        out[regime] = (alpha * local_bias + (1.0 - alpha) * global_bias).astype(np.float32)
    out -= out.mean(axis=1, keepdims=True)
    return out


def make_seq_features(x: np.ndarray, end_idx: int, direction: float = 1.0) -> np.ndarray:
    feats = []

    def velocity_at(idx: int) -> np.ndarray:
        if idx <= 0:
            raise ValueError("make_seq_features requires idx > 0.")
        return x[:, idx] - x[:, idx - 1]

    if end_idx < 3:
        raise ValueError("make_seq_features requires end_idx >= 3.")
    indices = list(range(max(3, end_idx - 5), end_idx + 1))
    if len(indices) < 6:
        indices = [indices[0]] * (6 - len(indices)) + indices
    for idx in indices:
        d1 = velocity_at(idx)
        d2 = velocity_at(idx - 1)
        d3 = velocity_at(idx - 2)
        acc = d1 - d2
        prev_acc = d2 - d3
        tangent = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + EPS)
        speed = np.linalg.norm(d1, axis=1, keepdims=True)
        prev_speed = np.linalg.norm(d2, axis=1, keepdims=True)
        acc_par_scalar = np.sum(acc * tangent, axis=1, keepdims=True)
        acc_par = acc_par_scalar * tangent
        acc_perp = acc - acc_par
        acc_norm = np.linalg.norm(acc, axis=1, keepdims=True)
        perp_norm = np.linalg.norm(acc_perp, axis=1, keepdims=True)
        jerk_norm = np.linalg.norm(acc - prev_acc, axis=1, keepdims=True)
        turn_cos = np.sum(d1 * d2, axis=1, keepdims=True) / ((speed * prev_speed) + EPS)
        curvature = perp_norm / (speed + EPS)
        raw_ctx = np.concatenate(
            [speed, prev_speed, acc_norm, acc_par_scalar, perp_norm, jerk_norm, turn_cos, curvature],
            axis=1,
        )
        feats.append(
            np.concatenate(
                [
                    turn_model_features_from_context(raw_ctx),
                    np.full((len(x), 1), direction, dtype=np.float32),
                ],
                axis=1,
            )
        )
    return np.stack(feats, axis=1).astype(np.float32)


def make_candidate_features(x: np.ndarray, end_idx: int, candidates: np.ndarray, horizon: int = 2, direction: float = 1.0) -> np.ndarray:
    p0, d1, acc = motion_terms(x, end_idx)
    tangent = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + EPS)
    delta = candidates - p0[:, None, :]
    par = np.sum(delta * tangent[:, None, :], axis=2, keepdims=True)
    perp = np.linalg.norm(delta - par * tangent[:, None, :], axis=2, keepdims=True)
    dist = np.linalg.norm(delta, axis=2, keepdims=True)
    speed = np.linalg.norm(d1, axis=1, keepdims=True)[:, None, :]
    scale = np.maximum(speed * float(horizon), EPS)
    spec = candidate_spec_features(len(x))
    ctx_base = np.concatenate(
        [
            turn_model_features_from_context(turn_context_features(x, end_idx)),
            np.full((len(x), 1), direction, dtype=np.float32),
        ],
        axis=1,
    )
    ctx = ctx_base[:, None, :].repeat(candidates.shape[1], axis=1)
    interactions = spec[:, :, 1:3] * ctx[:, :, [3, 4]]
    return np.concatenate(
        [
            par / scale,
            perp / scale,
            dist / scale,
            spec,
            ctx,
            interactions,
        ],
        axis=2,
    ).astype(np.float32)


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state_dict_cpu(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict({k: v.clone() for k, v in state.items()})


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    encoder = getattr(model, "gru", None)
    if encoder is None:
        encoder = getattr(model, "net", None)
    if encoder is None:
        return
    for param in encoder.parameters():
        param.requires_grad = trainable


def augment_trajectories_physics(x: np.ndarray, seed: int, speed_jitter: float, accel_noise: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = np.diff(x, axis=1)
    speed = np.linalg.norm(d, axis=2, keepdims=True)
    typical = np.median(speed, axis=1, keepdims=True)
    scale = rng.uniform(1.0 - speed_jitter, 1.0 + speed_jitter, size=(len(x), 1, 1)).astype(np.float32)
    noise = rng.normal(0.0, accel_noise, size=d.shape).astype(np.float32) * (typical + EPS)
    noise[:, 1:-1] = 0.25 * noise[:, :-2] + 0.5 * noise[:, 1:-1] + 0.25 * noise[:, 2:]
    d_aug = d * scale + noise
    x_aug = np.empty_like(x)
    x_aug[:, 0] = x[:, 0]
    x_aug[:, 1:] = x[:, 0:1] + np.cumsum(d_aug, axis=1)
    return x_aug.astype(np.float32)


def phase_shift_trajectories_physics(x: np.ndarray, phase: float) -> np.ndarray:
    t = np.arange(x.shape[1], dtype=np.float32) + float(phase)
    v = np.empty_like(x)
    v[:, 1:-1] = 0.5 * (x[:, 2:] - x[:, :-2])
    v[:, 0] = x[:, 1] - x[:, 0]
    v[:, -1] = x[:, -1] - x[:, -2]
    out = np.empty_like(x)
    for j, tj in enumerate(t):
        if tj <= 0:
            out[:, j] = x[:, 0] + tj * v[:, 0]
            continue
        if tj >= x.shape[1] - 1:
            out[:, j] = x[:, -1] + (tj - (x.shape[1] - 1)) * v[:, -1]
            continue
        i = int(np.floor(float(tj)))
        a = float(tj - i)
        h00 = 2.0 * a**3 - 3.0 * a**2 + 1.0
        h10 = a**3 - 2.0 * a**2 + a
        h01 = -2.0 * a**3 + 3.0 * a**2
        h11 = a**3 - a**2
        out[:, j] = h00 * x[:, i] + h10 * v[:, i] + h01 * x[:, i + 1] + h11 * v[:, i + 1]
    return out.astype(np.float32)


def mix_relative_physics_samples(
    seqs: np.ndarray,
    cand_feats: np.ndarray,
    labels: np.ndarray,
    seed: int,
    copies: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if copies <= 0:
        return seqs[:0], cand_feats[:0], labels[:0]
    rng = np.random.default_rng(seed)
    hard = np.argmax(labels, axis=1)
    mixed_seq, mixed_cf, mixed_label = [], [], []
    for _ in range(copies):
        partner = np.arange(len(labels))
        for cand_id in np.unique(hard):
            idx = np.flatnonzero(hard == cand_id)
            if len(idx) > 1:
                partner[idx] = rng.permutation(idx)
        lam = rng.beta(8.0, 8.0, size=(len(labels), 1)).astype(np.float32)
        seq_lam = lam[:, None, :]
        mixed_seq.append(seq_lam * seqs + (1.0 - seq_lam) * seqs[partner])
        mixed_cf.append(lam[:, None, :] * cand_feats + (1.0 - lam[:, None, :]) * cand_feats[partner])
        mixed_label.append(lam * labels + (1.0 - lam) * labels[partner])
    return (
        np.vstack(mixed_seq).astype(np.float32),
        np.vstack(mixed_cf).astype(np.float32),
        np.vstack(mixed_label).astype(np.float32),
    )


def strict_interpolation_mask(
    labels: np.ndarray,
    min_confidence: float,
    forbid_acc_shortcuts: bool,
) -> np.ndarray:
    best = np.argmax(labels, axis=1)
    confidence = np.max(labels, axis=1)
    mask = confidence >= min_confidence
    if forbid_acc_shortcuts:
        names = np.asarray([spec.name for spec in CANDIDATES], dtype=object)
        best_names = names[best]
        shortcut = np.char.startswith(best_names.astype(str), "acc_") | (best_names == "p0_2d1")
        mask &= ~shortcut
    return mask


def best_candidate_labels(candidates: np.ndarray, target: np.ndarray) -> np.ndarray:
    err = np.linalg.norm(candidates - target[:, None, :], axis=2)
    return np.argmin(err, axis=1).astype(np.int64)


def soft_candidate_targets(candidates: np.ndarray, target: np.ndarray, tau: float = 0.0045) -> np.ndarray:
    err = np.linalg.norm(candidates - target[:, None, :], axis=2)
    score = -err / tau
    score += (err <= R_HIT).astype(np.float32) * 0.75
    score -= score.max(axis=1, keepdims=True)
    prob = np.exp(score)
    prob /= prob.sum(axis=1, keepdims=True) + EPS
    return prob.astype(np.float32)


def label_confidence_weights(labels: np.ndarray, base: float = 0.6, scale: float = 2.4) -> np.ndarray:
    uniform = 1.0 / labels.shape[1]
    sharp = (np.max(labels, axis=1) - uniform) / (1.0 - uniform + EPS)
    return (base + scale * np.clip(sharp, 0.0, 1.0)).astype(np.float32)


def build_samples(
    x: np.ndarray,
    y: np.ndarray | None,
    final: bool,
    late_only: bool = False,
    horizons: tuple[int, ...] = (2,),
    direction: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seqs, cand_feats, cands, labels = [], [], [], []
    if final:
        assert y is not None
        end_targets = [(x.shape[1] - 1, 2, y)]
    else:
        end_targets = []
        for horizon in horizons:
            min_end = 3
            max_end = x.shape[1] - horizon - 1
            if max_end < min_end:
                continue
            if late_only:
                end_range = range(max(min_end, max_end - 3), max_end + 1)
            else:
                end_range = range(min_end, max_end + 1)
            end_targets.extend((end_idx, horizon, x[:, end_idx + horizon]) for end_idx in end_range)
    for end_idx, horizon, target in end_targets:
        cand = make_candidates(x, end_idx, horizon=horizon)
        seqs.append(make_seq_features(x, end_idx, direction=direction))
        cand_feats.append(make_candidate_features(x, end_idx, cand, horizon=horizon, direction=direction))
        cands.append(cand)
        labels.append(soft_candidate_targets(cand, target))
    return np.vstack(seqs), np.vstack(cand_feats), np.vstack(cands), np.vstack(labels)


class GRUSelector(nn.Module):
    def __init__(self, seq_dim: int, cand_dim: int, hidden: int, cand_count: int):
        super().__init__()
        self.gru = nn.GRU(seq_dim, hidden, batch_first=True, num_layers=2, dropout=0.08)
        self.head = nn.Sequential(
            nn.Linear(hidden + cand_dim, hidden),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, 1),
        )
        self.cand_count = cand_count

    def forward(self, seq: torch.Tensor, cand_feat: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(seq)
        ctx = h[-1]
        ctx = ctx[:, None, :].expand(-1, self.cand_count, -1)
        return self.head(torch.cat([ctx, cand_feat], dim=2)).squeeze(-1)


class CandidateAttentionGRUSelector(nn.Module):
    def __init__(self, seq_dim: int, cand_dim: int, hidden: int, cand_count: int):
        super().__init__()
        self.gru = nn.GRU(seq_dim, hidden, batch_first=True, num_layers=2, dropout=0.08)
        self.query = nn.Sequential(
            nn.Linear(cand_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.ctx_norm = nn.LayerNorm(hidden)
        self.event_norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + cand_dim, hidden),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, 1),
        )
        self.cand_count = cand_count
        self.scale = math.sqrt(float(hidden))

    def forward(self, seq: torch.Tensor, cand_feat: torch.Tensor) -> torch.Tensor:
        out, h = self.gru(seq)
        final_ctx = self.ctx_norm(h[-1])
        query = self.query(cand_feat)
        attn_logits = torch.einsum("bth,bch->bct", out, query) / self.scale
        attn = torch.softmax(attn_logits, dim=2)
        event_ctx = torch.einsum("bct,bth->bch", attn, out)
        event_ctx = self.event_norm(event_ctx)
        final_ctx = final_ctx[:, None, :].expand(-1, self.cand_count, -1)
        return self.head(torch.cat([final_ctx, event_ctx, cand_feat], dim=2)).squeeze(-1)


class LidarPhysicsGRUSelector(nn.Module):
    def __init__(self, seq_dim: int, cand_dim: int, hidden: int, cand_count: int):
        super().__init__()
        self.gru = nn.GRU(seq_dim, hidden, batch_first=True, num_layers=2, dropout=0.08)
        self.summary_proj = nn.Sequential(
            nn.Linear(seq_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
        )
        self.query = nn.Sequential(
            nn.Linear(cand_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.ctx_norm = nn.LayerNorm(hidden)
        self.token_norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + cand_dim, hidden),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, 1),
        )
        self.cand_count = cand_count
        self.scale = math.sqrt(float(hidden))

    def forward(self, seq: torch.Tensor, cand_feat: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(seq)
        final_ctx = self.ctx_norm(h[-1])

        mean = seq.mean(dim=1)
        std = seq.std(dim=1, unbiased=False)
        drift = seq[:, -1] - seq[:, 0]
        peak = seq.abs().amax(dim=1)
        tokens = torch.stack([mean, std, drift, peak], dim=1)
        tokens = self.summary_proj(tokens)

        query = self.query(cand_feat)
        attn_logits = torch.einsum("bkh,bch->bck", tokens, query) / self.scale
        attn = torch.softmax(attn_logits, dim=2)
        physics_ctx = torch.einsum("bck,bkh->bch", attn, tokens)
        physics_ctx = self.token_norm(physics_ctx)

        final_ctx = final_ctx[:, None, :].expand(-1, self.cand_count, -1)
        return self.head(torch.cat([final_ctx, physics_ctx, cand_feat], dim=2)).squeeze(-1)


class BiGRUSelector(nn.Module):
    def __init__(self, seq_dim: int, cand_dim: int, hidden: int, cand_count: int):
        super().__init__()
        self.gru = nn.GRU(seq_dim, hidden, batch_first=True, num_layers=2, dropout=0.08, bidirectional=True)
        ctx_dim = hidden * 2
        self.head = nn.Sequential(
            nn.Linear(ctx_dim + cand_dim, hidden),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, 1),
        )
        self.cand_count = cand_count

    def forward(self, seq: torch.Tensor, cand_feat: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(seq)
        ctx = torch.cat([h[-2], h[-1]], dim=1)
        ctx = ctx[:, None, :].expand(-1, self.cand_count, -1)
        return self.head(torch.cat([ctx, cand_feat], dim=2)).squeeze(-1)


class LSTMSelector(nn.Module):
    def __init__(self, seq_dim: int, cand_dim: int, hidden: int, cand_count: int):
        super().__init__()
        self.lstm = nn.LSTM(seq_dim, hidden, batch_first=True, num_layers=2, dropout=0.08)
        self.ctx_norm = nn.LayerNorm(hidden)
        self.cell_filter = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.ctx_dropout = nn.Dropout(0.12)
        self.head = nn.Sequential(
            nn.Linear(hidden + cand_dim, hidden),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, 1),
        )
        self.cand_count = cand_count

    def forward(self, seq: torch.Tensor, cand_feat: torch.Tensor) -> torch.Tensor:
        _, (h, c) = self.lstm(seq)
        ctx = self.ctx_norm(h[-1])
        ctx = ctx * (0.5 + 0.5 * self.cell_filter(c[-1]))
        ctx = self.ctx_dropout(ctx)
        ctx = ctx[:, None, :].expand(-1, self.cand_count, -1)
        return self.head(torch.cat([ctx, cand_feat], dim=2)).squeeze(-1)


class HierarchicalGRUSelector(nn.Module):
    def __init__(self, seq_dim: int, cand_dim: int, hidden: int, cand_count: int, family_scale: float):
        super().__init__()
        self.gru = nn.GRU(seq_dim, hidden, batch_first=True, num_layers=2, dropout=0.08)
        self.within_head = nn.Sequential(
            nn.Linear(hidden + cand_dim, hidden),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, 1),
        )
        self.family_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden // 2, len(FAMILY_NAMES)),
        )
        self.register_buffer("candidate_family", torch.from_numpy(CANDIDATE_FAMILY).long())
        self.cand_count = cand_count
        self.family_scale = float(family_scale)

    def forward(self, seq: torch.Tensor, cand_feat: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(seq)
        ctx = h[-1]
        ctx_rep = ctx[:, None, :].expand(-1, self.cand_count, -1)
        within = self.within_head(torch.cat([ctx_rep, cand_feat], dim=2)).squeeze(-1)
        family = self.family_head(ctx)
        return within + self.family_scale * family[:, self.candidate_family]


class TCNSelector(nn.Module):
    def __init__(self, seq_dim: int, cand_dim: int, hidden: int, cand_count: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConstantPad1d((2, 0), 0.0),
            nn.Conv1d(seq_dim, hidden, kernel_size=3, dilation=1),
            nn.SiLU(),
            nn.ConstantPad1d((4, 0), 0.0),
            nn.Conv1d(hidden, hidden, kernel_size=3, dilation=2),
            nn.SiLU(),
            nn.ConstantPad1d((8, 0), 0.0),
            nn.Conv1d(hidden, hidden, kernel_size=3, dilation=4),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + cand_dim, hidden),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, 1),
        )
        self.cand_count = cand_count

    def forward(self, seq: torch.Tensor, cand_feat: torch.Tensor) -> torch.Tensor:
        z = self.net(seq.transpose(1, 2))
        ctx = z[:, :, -1]
        ctx = ctx[:, None, :].expand(-1, self.cand_count, -1)
        return self.head(torch.cat([ctx, cand_feat], dim=2)).squeeze(-1)


class PhysicsGraphTCNSelector(nn.Module):
    def __init__(self, seq_dim: int, cand_dim: int, hidden: int, cand_count: int):
        super().__init__()
        node_hidden = max(16, hidden // 4)
        self.node_slices = [
            [0, 1],
            [2, 3, 4, 5],
            [6, 7],
            [8],
        ]
        self.node_proj = nn.ModuleList([nn.Sequential(nn.Linear(len(idx), node_hidden), nn.SiLU()) for idx in self.node_slices])
        self.node_tcn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConstantPad1d((2, 0), 0.0),
                    nn.Conv1d(node_hidden, node_hidden, kernel_size=3, dilation=1),
                    nn.SiLU(),
                    nn.ConstantPad1d((4, 0), 0.0),
                    nn.Conv1d(node_hidden, node_hidden, kernel_size=3, dilation=2),
                    nn.SiLU(),
                )
                for _ in self.node_slices
            ]
        )
        self.node_mix = nn.Parameter(torch.eye(len(self.node_slices), dtype=torch.float32))
        self.query = nn.Sequential(
            nn.Linear(cand_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, len(self.node_slices)),
        )
        self.ctx_norm = nn.LayerNorm(node_hidden * len(self.node_slices))
        self.node_norm = nn.LayerNorm(node_hidden)
        self.head = nn.Sequential(
            nn.Linear(node_hidden * (len(self.node_slices) + 1) + cand_dim, hidden),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, 1),
        )
        self.cand_count = cand_count

    def forward(self, seq: torch.Tensor, cand_feat: torch.Tensor) -> torch.Tensor:
        node_last = []
        for idx, proj, conv in zip(self.node_slices, self.node_proj, self.node_tcn):
            z = proj(seq[:, :, idx])
            z = conv(z.transpose(1, 2)).transpose(1, 2)
            node_last.append(z[:, -1])
        nodes = torch.stack(node_last, dim=1)
        mix = torch.softmax(self.node_mix, dim=1)
        nodes = torch.einsum("ij,bjh->bih", mix, nodes)
        global_ctx = self.ctx_norm(nodes.reshape(nodes.shape[0], -1))
        query = torch.softmax(self.query(cand_feat), dim=2)
        node_ctx = torch.einsum("bci,bih->bch", query, nodes)
        node_ctx = self.node_norm(node_ctx)
        global_ctx = global_ctx[:, None, :].expand(-1, self.cand_count, -1)
        return self.head(torch.cat([global_ctx, node_ctx, cand_feat], dim=2)).squeeze(-1)


class HierarchicalTCNSelector(nn.Module):
    def __init__(self, seq_dim: int, cand_dim: int, hidden: int, cand_count: int, family_scale: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConstantPad1d((2, 0), 0.0),
            nn.Conv1d(seq_dim, hidden, kernel_size=3, dilation=1),
            nn.SiLU(),
            nn.ConstantPad1d((4, 0), 0.0),
            nn.Conv1d(hidden, hidden, kernel_size=3, dilation=2),
            nn.SiLU(),
            nn.ConstantPad1d((8, 0), 0.0),
            nn.Conv1d(hidden, hidden, kernel_size=3, dilation=4),
            nn.SiLU(),
        )
        self.within_head = nn.Sequential(
            nn.Linear(hidden + cand_dim, hidden),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, 1),
        )
        self.family_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden // 2, len(FAMILY_NAMES)),
        )
        self.register_buffer("candidate_family", torch.from_numpy(CANDIDATE_FAMILY).long())
        self.cand_count = cand_count
        self.family_scale = float(family_scale)

    def forward(self, seq: torch.Tensor, cand_feat: torch.Tensor) -> torch.Tensor:
        z = self.net(seq.transpose(1, 2))
        ctx = z[:, :, -1]
        ctx_rep = ctx[:, None, :].expand(-1, self.cand_count, -1)
        within = self.within_head(torch.cat([ctx_rep, cand_feat], dim=2)).squeeze(-1)
        family = self.family_head(ctx)
        return within + self.family_scale * family[:, self.candidate_family]


class LatentEnvGRUSelector(nn.Module):
    def __init__(self, seq_dim: int, cand_dim: int, hidden: int, cand_count: int, experts: int, family_scale: float):
        super().__init__()
        self.gru = nn.GRU(seq_dim, hidden, batch_first=True, num_layers=2, dropout=0.08)
        self.expert_head = nn.Sequential(
            nn.Linear(hidden + cand_dim, hidden),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, experts),
        )
        self.env_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden // 2, experts),
        )
        self.family_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden // 2, len(FAMILY_NAMES)),
        )
        self.register_buffer("candidate_family", torch.from_numpy(CANDIDATE_FAMILY).long())
        self.cand_count = cand_count
        self.family_scale = float(family_scale)

    def forward(self, seq: torch.Tensor, cand_feat: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(seq)
        ctx = h[-1]
        ctx_rep = ctx[:, None, :].expand(-1, self.cand_count, -1)
        expert = self.expert_head(torch.cat([ctx_rep, cand_feat], dim=2))
        env_logp = torch.log_softmax(self.env_head(ctx), dim=1)
        mixed = torch.logsumexp(expert + env_logp[:, None, :], dim=2)
        family = self.family_head(ctx)
        return mixed + self.family_scale * family[:, self.candidate_family]


class LatentEnvTCNSelector(nn.Module):
    def __init__(self, seq_dim: int, cand_dim: int, hidden: int, cand_count: int, experts: int, family_scale: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConstantPad1d((2, 0), 0.0),
            nn.Conv1d(seq_dim, hidden, kernel_size=3, dilation=1),
            nn.SiLU(),
            nn.ConstantPad1d((4, 0), 0.0),
            nn.Conv1d(hidden, hidden, kernel_size=3, dilation=2),
            nn.SiLU(),
            nn.ConstantPad1d((8, 0), 0.0),
            nn.Conv1d(hidden, hidden, kernel_size=3, dilation=4),
            nn.SiLU(),
        )
        self.expert_head = nn.Sequential(
            nn.Linear(hidden + cand_dim, hidden),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, experts),
        )
        self.env_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden // 2, experts),
        )
        self.family_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden // 2, len(FAMILY_NAMES)),
        )
        self.register_buffer("candidate_family", torch.from_numpy(CANDIDATE_FAMILY).long())
        self.cand_count = cand_count
        self.family_scale = float(family_scale)

    def forward(self, seq: torch.Tensor, cand_feat: torch.Tensor) -> torch.Tensor:
        z = self.net(seq.transpose(1, 2))
        ctx = z[:, :, -1]
        ctx_rep = ctx[:, None, :].expand(-1, self.cand_count, -1)
        expert = self.expert_head(torch.cat([ctx_rep, cand_feat], dim=2))
        env_logp = torch.log_softmax(self.env_head(ctx), dim=1)
        mixed = torch.logsumexp(expert + env_logp[:, None, :], dim=2)
        family = self.family_head(ctx)
        return mixed + self.family_scale * family[:, self.candidate_family]


class LatentPhysicsAdapterMixin:
    def _init_physics_adapter(self, hidden: int, cand_dim: int, cand_count: int, family_scale: float, coeff_scale: float) -> None:
        self.residual_head = nn.Sequential(
            nn.Linear(hidden + cand_dim, hidden),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, 1),
        )
        self.coeff_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden // 2, CANDIDATE_SPEC_MATRIX.shape[1]),
        )
        self.family_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden // 2, len(FAMILY_NAMES)),
        )
        self.register_buffer("candidate_family", torch.from_numpy(CANDIDATE_FAMILY).long())
        self.register_buffer("candidate_spec", torch.from_numpy(CANDIDATE_SPEC_MATRIX).float())
        self.register_buffer("spec_center", torch.from_numpy(CANDIDATE_SPEC_MATRIX.mean(axis=0)).float())
        self.register_buffer("spec_scale", torch.from_numpy(CANDIDATE_SPEC_MATRIX.std(axis=0) + 1e-3).float())
        self.cand_count = cand_count
        self.family_scale = float(family_scale)
        self.coeff_scale = float(coeff_scale)

    def _physics_scores(self, ctx: torch.Tensor, cand_feat: torch.Tensor) -> torch.Tensor:
        ctx_rep = ctx[:, None, :].expand(-1, self.cand_count, -1)
        residual = self.residual_head(torch.cat([ctx_rep, cand_feat], dim=2)).squeeze(-1)
        coeff = self.spec_center + 0.35 * torch.tanh(self.coeff_head(ctx)) * self.spec_scale
        spec_dist = ((self.candidate_spec[None, :, :] - coeff[:, None, :]) / self.spec_scale[None, None, :]) ** 2
        physics = -self.coeff_scale * spec_dist.mean(dim=2)
        family = self.family_head(ctx)
        return physics + residual + self.family_scale * family[:, self.candidate_family]


class LatentPhysicsGRUSelector(nn.Module, LatentPhysicsAdapterMixin):
    def __init__(self, seq_dim: int, cand_dim: int, hidden: int, cand_count: int, family_scale: float, coeff_scale: float):
        super().__init__()
        self.gru = nn.GRU(seq_dim, hidden, batch_first=True, num_layers=2, dropout=0.08)
        self._init_physics_adapter(hidden, cand_dim, cand_count, family_scale, coeff_scale)

    def forward(self, seq: torch.Tensor, cand_feat: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(seq)
        return self._physics_scores(h[-1], cand_feat)


class LatentPhysicsBiGRUSelector(nn.Module, LatentPhysicsAdapterMixin):
    def __init__(self, seq_dim: int, cand_dim: int, hidden: int, cand_count: int, family_scale: float, coeff_scale: float):
        super().__init__()
        self.gru = nn.GRU(seq_dim, hidden, batch_first=True, num_layers=2, dropout=0.08, bidirectional=True)
        self._init_physics_adapter(hidden * 2, cand_dim, cand_count, family_scale, coeff_scale)

    def forward(self, seq: torch.Tensor, cand_feat: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(seq)
        ctx = torch.cat([h[-2], h[-1]], dim=1)
        return self._physics_scores(ctx, cand_feat)


class LatentPhysicsTCNSelector(nn.Module, LatentPhysicsAdapterMixin):
    def __init__(self, seq_dim: int, cand_dim: int, hidden: int, cand_count: int, family_scale: float, coeff_scale: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConstantPad1d((2, 0), 0.0),
            nn.Conv1d(seq_dim, hidden, kernel_size=3, dilation=1),
            nn.SiLU(),
            nn.ConstantPad1d((4, 0), 0.0),
            nn.Conv1d(hidden, hidden, kernel_size=3, dilation=2),
            nn.SiLU(),
            nn.ConstantPad1d((8, 0), 0.0),
            nn.Conv1d(hidden, hidden, kernel_size=3, dilation=4),
            nn.SiLU(),
        )
        self._init_physics_adapter(hidden, cand_dim, cand_count, family_scale, coeff_scale)

    def forward(self, seq: torch.Tensor, cand_feat: torch.Tensor) -> torch.Tensor:
        z = self.net(seq.transpose(1, 2))
        return self._physics_scores(z[:, :, -1], cand_feat)


def make_selector_model(
    model_name: str,
    seq_dim: int,
    cand_dim: int,
    hidden: int,
    cand_count: int,
    *,
    hier_family_gate: bool,
    family_gate_scale: float,
    latent_env_experts: int,
    latent_physics_adapter: bool,
    latent_physics_coeff_scale: float,
) -> nn.Module:
    if latent_physics_adapter:
        if model_name == "gru":
            return LatentPhysicsGRUSelector(seq_dim, cand_dim, hidden, cand_count, family_gate_scale, latent_physics_coeff_scale)
        if model_name == "bigru":
            return LatentPhysicsBiGRUSelector(seq_dim, cand_dim, hidden, cand_count, family_gate_scale, latent_physics_coeff_scale)
        if model_name == "lstm":
            return LSTMSelector(seq_dim, cand_dim, hidden, cand_count)
        return LatentPhysicsTCNSelector(seq_dim, cand_dim, hidden, cand_count, family_gate_scale, latent_physics_coeff_scale)
    if latent_env_experts > 1:
        if model_name == "gru":
            return LatentEnvGRUSelector(seq_dim, cand_dim, hidden, cand_count, latent_env_experts, family_gate_scale)
        return LatentEnvTCNSelector(seq_dim, cand_dim, hidden, cand_count, latent_env_experts, family_gate_scale)
    if model_name == "gru":
        if hier_family_gate:
            return HierarchicalGRUSelector(seq_dim, cand_dim, hidden, cand_count, family_gate_scale)
        return GRUSelector(seq_dim, cand_dim, hidden, cand_count)
    if model_name == "attn_gru":
        return CandidateAttentionGRUSelector(seq_dim, cand_dim, hidden, cand_count)
    if model_name == "lidar_gru":
        return LidarPhysicsGRUSelector(seq_dim, cand_dim, hidden, cand_count)
    if model_name == "bigru":
        return BiGRUSelector(seq_dim, cand_dim, hidden, cand_count)
    if model_name == "lstm":
        return LSTMSelector(seq_dim, cand_dim, hidden, cand_count)
    if model_name == "physics_tcn":
        return PhysicsGraphTCNSelector(seq_dim, cand_dim, hidden, cand_count)
    if hier_family_gate:
        return HierarchicalTCNSelector(seq_dim, cand_dim, hidden, cand_count, family_gate_scale)
    return TCNSelector(seq_dim, cand_dim, hidden, cand_count)


def normalize_fit(seqs: np.ndarray, cand_feats: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sm = seqs.reshape(-1, seqs.shape[-1]).mean(axis=0)
    ss = seqs.reshape(-1, seqs.shape[-1]).std(axis=0)
    cm = cand_feats.reshape(-1, cand_feats.shape[-1]).mean(axis=0)
    cs = cand_feats.reshape(-1, cand_feats.shape[-1]).std(axis=0)
    ss[ss < 1e-6] = 1.0
    cs[cs < 1e-6] = 1.0
    return sm.astype(np.float32), ss.astype(np.float32), cm.astype(np.float32), cs.astype(np.float32)


def normalize_apply(seqs: np.ndarray, cand_feats: np.ndarray, norm: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    sm, ss, cm, cs = norm
    return ((seqs - sm) / ss).astype(np.float32), ((cand_feats - cm) / cs).astype(np.float32)


def top_distribution(scores: np.ndarray, limit: int = 4) -> list[dict[str, object]]:
    top = np.argmax(scores, axis=1)
    counts = np.bincount(top, minlength=len(CANDIDATES))
    order = np.argsort(-counts)[:limit]
    return [{"name": CANDIDATES[int(i)].name, "count": int(counts[int(i)])} for i in order if counts[int(i)] > 0]


def candidate_physics_bias(candidates: np.ndarray, target: np.ndarray) -> np.ndarray:
    err = np.linalg.norm(candidates - target[:, None, :], axis=2)
    hit = np.mean(err <= R_HIT, axis=0)
    mean = np.mean(err, axis=0)
    bias = np.log(hit + 1e-4) - 18.0 * mean
    bias = bias - np.mean(bias)
    return bias.astype(np.float32)


def softmax_np(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = scores / max(float(temperature), 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return (p / (p.sum(axis=1, keepdims=True) + EPS)).astype(np.float32)


def train_one(
    model: nn.Module,
    data: tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
    epochs: int,
    lr: float,
    batch: int,
    seed: int,
    *,
    stage: str,
    gauge: dict[str, object] | None = None,
    bias: np.ndarray | None = None,
    gauge_bias: np.ndarray | None = None,
    log_every: int = 1,
    patience: int = 999999,
    min_epochs: int = 1,
    pairwise_loss_weight: float = 0.0,
    pairwise_margin: float = 0.10,
    pairwise_min_label_gap: float = 0.04,
    fine_distill_weight: float = 0.0,
) -> list[float]:
    if len(data) == 3:
        seqs, cand_feats, labels = data
        weights = np.ones(len(labels), dtype=np.float32)
        teacher = np.zeros_like(labels, dtype=np.float32)
    elif len(data) == 4:
        seqs, cand_feats, labels, weights = data
        teacher = np.zeros_like(labels, dtype=np.float32)
    else:
        seqs, cand_feats, labels, weights, teacher = data
    ds = TensorDataset(torch.from_numpy(seqs), torch.from_numpy(cand_feats), torch.from_numpy(labels), torch.from_numpy(weights), torch.from_numpy(teacher))
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, generator=gen)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    losses = []
    best_soft_hit = -1.0
    best_epoch = 0
    best_state = clone_state_dict(model)
    wait = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for seq, cand, label, weight, teacher_prob in loader:
            seq = seq.to(device)
            cand = cand.to(device)
            label = label.to(device)
            weight = weight.to(device)
            teacher_prob = teacher_prob.to(device)
            bias_t = None if bias is None else torch.from_numpy(bias).to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(seq, cand)
            if bias_t is not None:
                logits = logits + bias_t[None, :]
            logp = torch.log_softmax(logits, dim=1)
            row_loss = -(label * logp).sum(dim=1)
            if fine_distill_weight > 0.0 and torch.any(teacher_prob > 0):
                teacher_loss = -(teacher_prob * logp).sum(dim=1)
                row_loss = (1.0 - fine_distill_weight) * row_loss + fine_distill_weight * teacher_loss
            if pairwise_loss_weight > 0.0:
                pos_idx = torch.argmax(label, dim=1)
                pos_label = label.gather(1, pos_idx[:, None]).squeeze(1)
                pos_score = logits.gather(1, pos_idx[:, None]).squeeze(1)
                neg_mask = label < (pos_label[:, None] - pairwise_min_label_gap)
                if torch.any(neg_mask):
                    neg_score = logits.masked_fill(~neg_mask, -1.0e9).max(dim=1).values
                    valid = neg_score > -1.0e8
                    pair_loss = torch.relu(pairwise_margin - pos_score + neg_score)
                    row_loss = row_loss + pairwise_loss_weight * pair_loss * valid.float()
            loss = (row_loss * weight).sum() / (weight.sum() + 1e-8)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(seq)
            n += len(seq)
        epoch_loss = total / max(n, 1)
        losses.append(epoch_loss)
        current_soft_hit = None
        if gauge is not None and (epoch == 1 or epoch == epochs or epoch % log_every == 0):
            val_scores = predict_scores(model, gauge["seq"], gauge["cand_feat"], device, batch)
            if gauge_bias is not None:
                val_scores = val_scores + gauge_bias
            elif bias is not None:
                val_scores = val_scores + bias[None, :]
            val_cands = gauge["candidates"]
            val_true = gauge["true"]
            argmax_pred = val_cands[np.arange(len(val_cands)), np.argmax(val_scores, axis=1)]
            arg_m = metrics(argmax_pred, val_true)
            temp_m = search_temperature(val_cands, val_scores, val_true)
            soft_m = temp_m["metrics"]
            current_soft_hit = float(soft_m["hit"])
            if current_soft_hit > best_soft_hit:
                best_soft_hit = current_soft_hit
                best_epoch = epoch
                best_state = clone_state_dict(model)
                wait = 0
            else:
                wait += 1
            print(
                "GAUGE",
                f"model={gauge['model']}",
                f"fold={gauge['fold']}",
                f"stage={stage}",
                f"epoch={epoch}/{epochs}",
                f"loss={epoch_loss:.5f}",
                f"arg_hit={arg_m['hit']:.4f}",
                f"soft_hit={soft_m['hit']:.4f}",
                f"soft_temp={temp_m['temperature']}",
                f"arg_mean={arg_m['mean']:.6f}",
                f"soft_mean={soft_m['mean']:.6f}",
                f"best_soft_hit={best_soft_hit:.4f}",
                f"wait={wait}/{patience}",
                f"top={top_distribution(val_scores)}",
                flush=True,
            )
        elif gauge is None and (epoch == 1 or epoch == epochs or epoch % log_every == 0):
            print(
                "FULL_GAUGE",
                f"stage={stage}",
                f"epoch={epoch}/{epochs}",
                f"loss={epoch_loss:.5f}",
                flush=True,
            )
        if gauge is not None and epoch >= min_epochs and wait >= patience:
            print(
                "EARLY_STOP",
                f"model={gauge['model']}",
                f"fold={gauge['fold']}",
                f"stage={stage}",
                f"epoch={epoch}",
                f"best_soft_hit={best_soft_hit:.4f}",
                flush=True,
            )
            break
    if gauge is not None:
        load_state_dict_cpu(model, best_state)
    model._last_best_epoch = int(best_epoch)
    model._last_best_hit = float(best_soft_hit)
    return losses


@torch.no_grad()
def predict_scores(model: nn.Module, seqs: np.ndarray, cand_feats: np.ndarray, device: torch.device, batch: int) -> np.ndarray:
    model.eval()
    outs = []
    for start in range(0, len(seqs), batch):
        seq = torch.from_numpy(seqs[start : start + batch]).to(device)
        cand = torch.from_numpy(cand_feats[start : start + batch]).to(device)
        outs.append(model(seq, cand).detach().cpu().numpy())
    return np.vstack(outs)


def soft_select(candidates: np.ndarray, scores: np.ndarray, temperature: float) -> np.ndarray:
    z = scores / max(temperature, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    w = np.exp(z)
    w = w / (w.sum(axis=1, keepdims=True) + EPS)
    return np.sum(candidates * w[:, :, None], axis=1)


def search_temperature(candidates: np.ndarray, scores: np.ndarray, true: np.ndarray) -> dict[str, object]:
    best = None
    best_key = (-1, -float("inf"))
    for temp in [0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.00]:
        pred = soft_select(candidates, scores, temp)
        m = metrics(pred, true)
        key = (int(m["hits"]), -float(m["mean"]))
        if best is None or key > best_key:
            best = {"temperature": temp, "metrics": m}
            best_key = key
    assert best is not None
    return best


def search_physics_projection(
    candidates: np.ndarray,
    learned_scores: np.ndarray,
    prior_scores: np.ndarray,
    true: np.ndarray,
) -> dict[str, object]:
    best = None
    best_key = (-1, -float("inf"))
    residual = learned_scores - prior_scores
    for residual_weight in [0.00, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 0.90, 1.00, 1.10, 1.25]:
        projected = prior_scores + residual_weight * residual
        temp_m = search_temperature(candidates, projected, true)
        m = temp_m["metrics"]
        key = (int(m["hits"]), -float(m["mean"]))
        if best is None or key > best_key:
            best = {
                "residual_weight": residual_weight,
                "temperature": temp_m["temperature"],
                "metrics": m,
            }
            best_key = key
    assert best is not None
    return best


def search_argmax_soft_gate(candidates: np.ndarray, scores: np.ndarray, true: np.ndarray) -> dict[str, object]:
    best = None
    best_key = (-1, -float("inf"))
    order = np.argsort(scores, axis=1)
    top1 = order[:, -1]
    top2 = order[:, -2]
    margin = scores[np.arange(len(scores)), top1] - scores[np.arange(len(scores)), top2]
    arg_pred = candidates[np.arange(len(candidates)), top1]
    for temp in [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
        soft_pred = soft_select(candidates, scores, temp)
        for q in [0.00, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.98, 1.01]:
            threshold = float(np.quantile(margin, q)) if q <= 1.0 else float("inf")
            use_arg = margin >= threshold
            pred = np.where(use_arg[:, None], arg_pred, soft_pred)
            m = metrics(pred, true)
            key = (int(m["hits"]), -float(m["mean"]))
            if best is None or key > best_key:
                best = {
                    "temperature": temp,
                    "margin_quantile": q,
                    "margin_threshold": threshold,
                    "argmax_rate": float(np.mean(use_arg)),
                    "metrics": m,
                }
                best_key = key
    assert best is not None
    return best


def evaluate_selector_state(
    model: nn.Module,
    gauge: dict[str, object],
    device: torch.device,
    batch: int,
    bias: np.ndarray,
) -> dict[str, float | int]:
    scores = predict_scores(model, gauge["seq"], gauge["cand_feat"], device, batch) + bias
    cands = gauge["candidates"]
    true = gauge["true"]
    pred = soft_select(cands, scores, 0.10)
    return metrics(pred, true)


def write_submission(path: Path, ids: list[str], pred: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "x", "y", "z"])
        for sample_id, row in zip(ids, pred):
            writer.writerow([sample_id, f"{row[0]:.9f}", f"{row[1]:.9f}", f"{row[2]:.9f}"])


def run_fold(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    train_ids: list[str],
    fold: int,
    folds: int,
    args: argparse.Namespace,
    device: torch.device,
):
    fold_ids = np.asarray([stable_fold_id(sample_id, folds) for sample_id in train_ids])
    va = fold_ids == fold
    tr = ~va
    pre_seq, pre_cf, _, pre_label = build_samples(train_x[tr], None, final=False, late_only=False, horizons=(1, 2))
    real_pre_seq, real_pre_cf = pre_seq.copy(), pre_cf.copy()
    pre_weight = np.ones(len(pre_label), dtype=np.float32)
    if args.reverse_pretrain:
        rev_seq, rev_cf, _, rev_label = build_samples(
            train_x[tr, ::-1].copy(),
            None,
            final=False,
            late_only=False,
            horizons=(1, 2),
            direction=-1.0,
        )
        pre_seq = np.vstack([pre_seq, rev_seq])
        pre_cf = np.vstack([pre_cf, rev_cf])
        pre_label = np.vstack([pre_label, rev_label])
        pre_weight = np.concatenate([pre_weight, np.full(len(rev_label), args.reverse_pretrain_weight, dtype=np.float32)])
    for phase in args.interp_phases:
        phase_x = phase_shift_trajectories_physics(train_x[tr], -phase)
        phase_seq, phase_cf, _, phase_label = build_samples(phase_x, None, final=False, late_only=False, horizons=(1, 2))
        phase_mask = strict_interpolation_mask(
            phase_label,
            min_confidence=args.interp_min_confidence,
            forbid_acc_shortcuts=args.interp_forbid_acc_shortcuts,
        )
        phase_seq, phase_cf, phase_label = phase_seq[phase_mask], phase_cf[phase_mask], phase_label[phase_mask]
        if len(phase_label) == 0:
            continue
        pre_seq = np.vstack([pre_seq, phase_seq])
        pre_cf = np.vstack([pre_cf, phase_cf])
        pre_label = np.vstack([pre_label, phase_label])
        pre_weight = np.concatenate([pre_weight, np.full(len(phase_label), args.interp_weight, dtype=np.float32)])
    for aug_idx in range(args.augment_copies):
        aug_x = augment_trajectories_physics(
            train_x[tr],
            seed=args.seed + fold * 100 + aug_idx,
            speed_jitter=args.augment_speed_jitter,
            accel_noise=args.augment_accel_noise,
        )
        aug_seq, aug_cf, _, aug_label = build_samples(aug_x, None, final=False, late_only=False, horizons=(1, 2))
        pre_seq = np.vstack([pre_seq, aug_seq])
        pre_cf = np.vstack([pre_cf, aug_cf])
        pre_label = np.vstack([pre_label, aug_label])
        pre_weight = np.concatenate([pre_weight, np.full(len(aug_label), args.augment_weight, dtype=np.float32)])
    if args.mix_copies > 0:
        mix_seq, mix_cf, mix_label = mix_relative_physics_samples(
            pre_seq,
            pre_cf,
            pre_label,
            seed=args.seed + fold * 409 + 77,
            copies=args.mix_copies,
        )
        pre_seq = np.vstack([pre_seq, mix_seq])
        pre_cf = np.vstack([pre_cf, mix_cf])
        pre_label = np.vstack([pre_label, mix_label])
        pre_weight = np.concatenate([pre_weight, np.full(len(mix_label), args.mix_weight, dtype=np.float32)])
    late_seq, late_cf, _, late_label = build_samples(train_x[tr], None, final=False, late_only=True, horizons=(1, 2))
    fin_seq, fin_cf, _, fin_label = build_samples(train_x[tr], train_y[tr], final=True)
    fine_seq_parts = [fin_seq] * args.final_repeat
    fine_cf_parts = [fin_cf] * args.final_repeat
    fine_label_parts = [fin_label] * args.final_repeat
    if args.include_late_fine:
        fine_seq_parts.insert(0, late_seq)
        fine_cf_parts.insert(0, late_cf)
        fine_label_parts.insert(0, late_label)
    fine_seq_raw = np.vstack(fine_seq_parts)
    fine_cf_raw = np.vstack(fine_cf_parts)
    fine_label = np.vstack(fine_label_parts)
    fine_weight = label_confidence_weights(fine_label)
    if args.norm_real_only:
        norm = normalize_fit(np.vstack([real_pre_seq, fine_seq_raw]), np.vstack([real_pre_cf, fine_cf_raw]))
    else:
        norm = normalize_fit(np.vstack([pre_seq, fine_seq_raw]), np.vstack([pre_cf, fine_cf_raw]))
    pre_seq, pre_cf = normalize_apply(pre_seq, pre_cf, norm)
    fine_seq, fine_cf = normalize_apply(fine_seq_raw, fine_cf_raw, norm)
    cand_count = len(CANDIDATES)
    model_offset = {"gru": 0, "attn_gru": 127, "lidar_gru": 191, "bigru": 251, "lstm": 379, "tcn": 503, "physics_tcn": 641}.get(model_name, 0)
    set_torch_seed(args.seed + fold * 1009 + model_offset)
    model = make_selector_model(
        model_name,
        pre_seq.shape[-1],
        pre_cf.shape[-1],
        args.hidden,
        cand_count,
        hier_family_gate=args.hier_family_gate,
        family_gate_scale=args.family_gate_scale,
        latent_env_experts=args.latent_env_experts,
        latent_physics_adapter=args.latent_physics_adapter,
        latent_physics_coeff_scale=args.latent_physics_coeff_scale,
    )
    model.to(device)
    val_seq, val_cf, val_cands, _ = build_samples(train_x[va], train_y[va], final=True)
    val_seq, val_cf = normalize_apply(val_seq, val_cf, norm)
    train_final_cands = make_candidates(train_x[tr], train_x.shape[1] - 1, horizon=2)
    physics_bias = candidate_physics_bias(train_final_cands, train_y[tr]) * args.prior_strength
    regime_bins = fit_regime_bins(train_x[tr], train_x.shape[1] - 1)
    train_regimes = assign_regimes(train_x[tr], train_x.shape[1] - 1, regime_bins)
    val_regimes = assign_regimes(train_x[va], train_x.shape[1] - 1, regime_bins)
    regime_table = candidate_regime_bias(train_final_cands, train_y[tr], train_regimes, regime_count=18)
    train_bias_final = physics_bias[None, :] + args.regime_prior_strength * regime_table[train_regimes]
    fine_bias_parts = []
    if args.include_late_fine:
        fine_bias_parts.append(np.tile(physics_bias[None, :], (len(late_label), 1)))
    fine_bias_parts.extend([train_bias_final] * args.final_repeat)
    fine_bias = np.vstack(fine_bias_parts).astype(np.float32)
    val_bias = physics_bias[None, :] + args.regime_prior_strength * regime_table[val_regimes]
    oracle = metrics(val_cands[np.arange(len(val_cands)), best_candidate_labels(val_cands, train_y[va])], train_y[va])
    print(
        "FOLD_START",
        f"model={model_name}",
        f"fold={fold + 1}/{folds}",
        f"train_pre={len(pre_label)}",
        f"pre_eff={float(pre_weight.sum()):.1f}",
        f"norm={'real' if args.norm_real_only else 'all'}",
        f"train_fine={len(fine_label)}",
        f"val={int(np.sum(va))}",
        f"oracle_hit={oracle['hit']:.4f}",
        f"prior_top={top_distribution(val_bias)}",
        flush=True,
    )
    gauge = {
        "model": model_name,
        "fold": fold + 1,
        "seq": val_seq,
        "cand_feat": val_cf,
        "candidates": val_cands,
        "true": train_y[va],
    }
    pre_losses = train_one(
        model,
        (pre_seq, pre_cf, pre_label, pre_weight),
        device,
        args.pre_epochs,
        args.lr,
        args.batch,
        args.seed + fold * 31,
        stage="pretrain",
        gauge=gauge,
        bias=physics_bias,
        gauge_bias=val_bias,
        log_every=args.log_every,
        patience=args.patience,
        min_epochs=args.min_epochs,
        pairwise_loss_weight=args.pairwise_loss_weight,
        pairwise_margin=args.pairwise_margin,
        pairwise_min_label_gap=args.pairwise_min_label_gap,
    )
    pre_state = clone_state_dict(model)
    pre_metric = evaluate_selector_state(model, gauge, device, args.batch, val_bias)
    pre_best_epoch = int(getattr(model, "_last_best_epoch", 0))
    pre_best_hit = float(getattr(model, "_last_best_hit", -1.0))
    fine_teacher = np.zeros_like(fine_label, dtype=np.float32)
    if args.fine_distill_weight > 0.0:
        teacher_scores = predict_scores(model, fine_seq, fine_cf, device, args.batch)
        if len(fine_bias) == len(teacher_scores):
            teacher_scores = teacher_scores + fine_bias
        fine_teacher = softmax_np(teacher_scores, args.fine_distill_temp)
    fine_losses = []
    if args.fine_ensemble:
        branch_scores = []
        if args.include_pre_branch:
            branch_scores.append(predict_scores(model, val_seq, val_cf, device, args.batch) + val_bias)
        for branch_idx, lr_mult in enumerate(args.fine_branch_scales):
            branch = copy.deepcopy(model)
            load_state_dict_cpu(branch, pre_state)
            branch.to(device)
            branch_weight = fine_weight ** float(args.fine_branch_weight_powers[branch_idx % len(args.fine_branch_weight_powers)])
            branch_losses = []
            if args.freeze_fine_epochs > 0:
                set_encoder_trainable(branch, False)
                branch_losses.extend(
                    train_one(
                        branch,
                        (fine_seq, fine_cf, fine_label, branch_weight, fine_teacher),
                        device,
                        min(args.freeze_fine_epochs, args.fine_epochs),
                        args.lr * args.fine_lr_scale * lr_mult,
                        args.batch,
                        args.seed + fold * 31 + 5000 + branch_idx * 97,
                        stage=f"finetune_branch{branch_idx}_head",
                        gauge=gauge,
                        bias=physics_bias,
                        gauge_bias=val_bias,
                        log_every=args.log_every,
                        patience=args.patience,
                        min_epochs=min(args.min_epochs, args.freeze_fine_epochs),
                        pairwise_loss_weight=args.pairwise_loss_weight,
                        pairwise_margin=args.pairwise_margin,
                        pairwise_min_label_gap=args.pairwise_min_label_gap,
                        fine_distill_weight=args.fine_distill_weight,
                    )
                )
                set_encoder_trainable(branch, True)
            remain_epochs = max(0, args.fine_epochs - args.freeze_fine_epochs)
            if remain_epochs > 0:
                branch_losses.extend(
                    train_one(
                        branch,
                        (fine_seq, fine_cf, fine_label, branch_weight, fine_teacher),
                        device,
                        remain_epochs,
                        args.lr * args.fine_lr_scale * lr_mult * 0.55,
                        args.batch,
                        args.seed + fold * 31 + 7000 + branch_idx * 97,
                        stage=f"finetune_branch{branch_idx}",
                        gauge=gauge,
                        bias=physics_bias,
                        gauge_bias=val_bias,
                        log_every=args.log_every,
                        patience=args.patience,
                        min_epochs=args.min_epochs,
                        pairwise_loss_weight=args.pairwise_loss_weight,
                        pairwise_margin=args.pairwise_margin,
                        pairwise_min_label_gap=args.pairwise_min_label_gap,
                        fine_distill_weight=args.fine_distill_weight,
                    )
                )
            scores_b = predict_scores(branch, val_seq, val_cf, device, args.batch) + val_bias
            branch_scores.append(scores_b)
            branch_m = search_argmax_soft_gate(val_cands, scores_b, train_y[va])["metrics"]
            print(
                "FINE_BRANCH_RESULT",
                f"model={model_name}",
                f"fold={fold + 1}",
                f"branch={branch_idx}",
                f"lr_mult={lr_mult}",
                f"weight_power={args.fine_branch_weight_powers[branch_idx % len(args.fine_branch_weight_powers)]}",
                f"gate_hit={branch_m['hit']:.4f}",
                flush=True,
            )
            if branch_losses:
                fine_losses.extend(branch_losses[-1:])
        scores = np.mean(np.stack(branch_scores, axis=0), axis=0).astype(np.float32)
        ens_m = search_argmax_soft_gate(val_cands, scores, train_y[va])["metrics"]
        print(
            "FINE_BRANCH_ENSEMBLE",
            f"model={model_name}",
            f"fold={fold + 1}",
            f"branches={len(branch_scores)}",
            f"gate_hit={ens_m['hit']:.4f}",
            flush=True,
        )
        meta = {
            "pre_best_epoch": pre_best_epoch,
            "pre_best_hit": pre_best_hit,
            "fine_best_epoch": 0,
            "fine_best_hit": -1.0,
        }
        return va, val_cands, scores, val_bias, pre_losses[-1], fine_losses[-1] if fine_losses else pre_losses[-1], oracle, meta
    if args.freeze_fine_epochs > 0:
        set_encoder_trainable(model, False)
        fine_losses.extend(
            train_one(
                model,
                (fine_seq, fine_cf, fine_label, fine_weight, fine_teacher),
                device,
                min(args.freeze_fine_epochs, args.fine_epochs),
                args.lr * args.fine_lr_scale,
                args.batch,
                args.seed + fold * 31 + 1000,
                stage="finetune_head",
                gauge=gauge,
                bias=physics_bias,
                gauge_bias=val_bias,
                log_every=args.log_every,
                patience=args.patience,
                min_epochs=min(args.min_epochs, args.freeze_fine_epochs),
                pairwise_loss_weight=args.pairwise_loss_weight,
                pairwise_margin=args.pairwise_margin,
                pairwise_min_label_gap=args.pairwise_min_label_gap,
                fine_distill_weight=args.fine_distill_weight,
            )
        )
        set_encoder_trainable(model, True)
    remain_epochs = max(0, args.fine_epochs - args.freeze_fine_epochs)
    if remain_epochs > 0:
        fine_losses.extend(
            train_one(
                model,
                (fine_seq, fine_cf, fine_label, fine_weight, fine_teacher),
                device,
                remain_epochs,
                args.lr * args.fine_lr_scale * 0.55,
                args.batch,
                args.seed + fold * 31 + 2000,
                stage="finetune",
                gauge=gauge,
                bias=physics_bias,
                gauge_bias=val_bias,
                log_every=args.log_every,
                patience=args.patience,
                min_epochs=args.min_epochs,
                pairwise_loss_weight=args.pairwise_loss_weight,
                pairwise_margin=args.pairwise_margin,
                pairwise_min_label_gap=args.pairwise_min_label_gap,
                fine_distill_weight=args.fine_distill_weight,
            )
        )
    fine_metric = evaluate_selector_state(model, gauge, device, args.batch, val_bias)
    if int(fine_metric["hits"]) < int(pre_metric["hits"]):
        load_state_dict_cpu(model, pre_state)
        print(
            "FINETUNE_ROLLBACK",
            f"model={model_name}",
            f"fold={fold + 1}",
            f"pre_soft_hit={pre_metric['hit']:.4f}",
            f"fine_soft_hit={fine_metric['hit']:.4f}",
            flush=True,
        )
    scores = predict_scores(model, val_seq, val_cf, device, args.batch) + val_bias
    meta = {
        "pre_best_epoch": pre_best_epoch,
        "pre_best_hit": pre_best_hit,
        "fine_best_epoch": int(getattr(model, "_last_best_epoch", 0)) if fine_losses else 0,
        "fine_best_hit": float(getattr(model, "_last_best_hit", -1.0)) if fine_losses else -1.0,
    }
    return va, val_cands, scores, val_bias, pre_losses[-1], fine_losses[-1] if fine_losses else pre_losses[-1], oracle, meta


def train_full_predict(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    *,
    pre_epochs: int | None = None,
    fine_epochs: int | None = None,
):
    pre_seq, pre_cf, _, pre_label = build_samples(train_x, None, final=False, late_only=False, horizons=(1, 2))
    real_pre_seq, real_pre_cf = pre_seq.copy(), pre_cf.copy()
    pre_weight = np.ones(len(pre_label), dtype=np.float32)
    if args.reverse_pretrain:
        rev_seq, rev_cf, _, rev_label = build_samples(
            train_x[:, ::-1].copy(),
            None,
            final=False,
            late_only=False,
            horizons=(1, 2),
            direction=-1.0,
        )
        pre_seq = np.vstack([pre_seq, rev_seq])
        pre_cf = np.vstack([pre_cf, rev_cf])
        pre_label = np.vstack([pre_label, rev_label])
        pre_weight = np.concatenate([pre_weight, np.full(len(rev_label), args.reverse_pretrain_weight, dtype=np.float32)])
    for phase in args.interp_phases:
        phase_x = phase_shift_trajectories_physics(train_x, -phase)
        phase_seq, phase_cf, _, phase_label = build_samples(phase_x, None, final=False, late_only=False, horizons=(1, 2))
        phase_mask = strict_interpolation_mask(
            phase_label,
            min_confidence=args.interp_min_confidence,
            forbid_acc_shortcuts=args.interp_forbid_acc_shortcuts,
        )
        phase_seq, phase_cf, phase_label = phase_seq[phase_mask], phase_cf[phase_mask], phase_label[phase_mask]
        if len(phase_label) == 0:
            continue
        pre_seq = np.vstack([pre_seq, phase_seq])
        pre_cf = np.vstack([pre_cf, phase_cf])
        pre_label = np.vstack([pre_label, phase_label])
        pre_weight = np.concatenate([pre_weight, np.full(len(phase_label), args.interp_weight, dtype=np.float32)])
    for aug_idx in range(args.augment_copies):
        aug_x = augment_trajectories_physics(
            train_x,
            seed=args.seed + 99900 + aug_idx,
            speed_jitter=args.augment_speed_jitter,
            accel_noise=args.augment_accel_noise,
        )
        aug_seq, aug_cf, _, aug_label = build_samples(aug_x, None, final=False, late_only=False, horizons=(1, 2))
        pre_seq = np.vstack([pre_seq, aug_seq])
        pre_cf = np.vstack([pre_cf, aug_cf])
        pre_label = np.vstack([pre_label, aug_label])
        pre_weight = np.concatenate([pre_weight, np.full(len(aug_label), args.augment_weight, dtype=np.float32)])
    if args.mix_copies > 0:
        mix_seq, mix_cf, mix_label = mix_relative_physics_samples(
            pre_seq,
            pre_cf,
            pre_label,
            seed=args.seed + 990077,
            copies=args.mix_copies,
        )
        pre_seq = np.vstack([pre_seq, mix_seq])
        pre_cf = np.vstack([pre_cf, mix_cf])
        pre_label = np.vstack([pre_label, mix_label])
        pre_weight = np.concatenate([pre_weight, np.full(len(mix_label), args.mix_weight, dtype=np.float32)])
    late_seq, late_cf, _, late_label = build_samples(train_x, None, final=False, late_only=True, horizons=(1, 2))
    fin_seq, fin_cf, _, fin_label = build_samples(train_x, train_y, final=True)
    fine_seq_parts = [fin_seq] * args.final_repeat
    fine_cf_parts = [fin_cf] * args.final_repeat
    fine_label_parts = [fin_label] * args.final_repeat
    if args.include_late_fine:
        fine_seq_parts.insert(0, late_seq)
        fine_cf_parts.insert(0, late_cf)
        fine_label_parts.insert(0, late_label)
    fine_seq_raw = np.vstack(fine_seq_parts)
    fine_cf_raw = np.vstack(fine_cf_parts)
    fine_label = np.vstack(fine_label_parts)
    fine_weight = label_confidence_weights(fine_label)
    if args.norm_real_only:
        norm = normalize_fit(np.vstack([real_pre_seq, fine_seq_raw]), np.vstack([real_pre_cf, fine_cf_raw]))
    else:
        norm = normalize_fit(np.vstack([pre_seq, fine_seq_raw]), np.vstack([pre_cf, fine_cf_raw]))
    pre_seq, pre_cf = normalize_apply(pre_seq, pre_cf, norm)
    fine_seq, fine_cf = normalize_apply(fine_seq_raw, fine_cf_raw, norm)
    model_offset = {"gru": 0, "attn_gru": 127, "lidar_gru": 191, "bigru": 251, "lstm": 379, "tcn": 503, "physics_tcn": 641}.get(model_name, 0)
    set_torch_seed(args.seed + 900001 + model_offset)
    model = make_selector_model(
        model_name,
        pre_seq.shape[-1],
        pre_cf.shape[-1],
        args.hidden,
        len(CANDIDATES),
        hier_family_gate=args.hier_family_gate,
        family_gate_scale=args.family_gate_scale,
        latent_env_experts=args.latent_env_experts,
        latent_physics_adapter=args.latent_physics_adapter,
        latent_physics_coeff_scale=args.latent_physics_coeff_scale,
    )
    model.to(device)
    train_final_cands = make_candidates(train_x, train_x.shape[1] - 1, horizon=2)
    physics_bias = candidate_physics_bias(train_final_cands, train_y) * args.prior_strength
    regime_bins = fit_regime_bins(train_x, train_x.shape[1] - 1)
    train_regimes = assign_regimes(train_x, train_x.shape[1] - 1, regime_bins)
    regime_table = candidate_regime_bias(train_final_cands, train_y, train_regimes, regime_count=18)
    print("FULL_START", f"model={model_name}", f"pre={len(pre_label)}", f"pre_eff={float(pre_weight.sum()):.1f}", f"norm={'real' if args.norm_real_only else 'all'}", f"fine={len(fine_label)}", f"prior_top={top_distribution(physics_bias[None, :])}", flush=True)
    pre_losses = train_one(
        model,
        (pre_seq, pre_cf, pre_label, pre_weight),
        device,
        int(pre_epochs or args.pre_epochs),
        args.lr,
        args.batch,
        args.seed + 999,
        stage="pretrain_full",
        bias=physics_bias,
        log_every=args.log_every,
        pairwise_loss_weight=args.pairwise_loss_weight,
        pairwise_margin=args.pairwise_margin,
        pairwise_min_label_gap=args.pairwise_min_label_gap,
    )
    fine_teacher = np.zeros_like(fine_label, dtype=np.float32)
    if args.fine_distill_weight > 0.0:
        train_bias = physics_bias[None, :] + args.regime_prior_strength * regime_table[train_regimes]
        fine_bias_parts = []
        if args.include_late_fine:
            fine_bias_parts.append(np.tile(physics_bias[None, :], (len(late_label), 1)))
        fine_bias_parts.extend([train_bias] * args.final_repeat)
        fine_bias = np.vstack(fine_bias_parts).astype(np.float32)
        teacher_scores = predict_scores(model, fine_seq, fine_cf, device, args.batch)
        if len(fine_bias) == len(teacher_scores):
            teacher_scores = teacher_scores + fine_bias
        fine_teacher = softmax_np(teacher_scores, args.fine_distill_temp)
    fine_losses = train_one(
        model,
        (fine_seq, fine_cf, fine_label, fine_weight, fine_teacher),
        device,
        int(fine_epochs or args.fine_epochs),
        args.lr * args.fine_lr_scale,
        args.batch,
        args.seed + 1999,
        stage="finetune_full",
        bias=physics_bias,
        log_every=args.log_every,
        pairwise_loss_weight=args.pairwise_loss_weight,
        pairwise_margin=args.pairwise_margin,
        pairwise_min_label_gap=args.pairwise_min_label_gap,
        fine_distill_weight=args.fine_distill_weight,
    )
    test_seq = make_seq_features(test_x, test_x.shape[1] - 1)
    test_cands = make_candidates(test_x, test_x.shape[1] - 1, horizon=2)
    test_cf = make_candidate_features(test_x, test_x.shape[1] - 1, test_cands, horizon=2)
    test_seq, test_cf = normalize_apply(test_seq, test_cf, norm)
    test_regimes = assign_regimes(test_x, test_x.shape[1] - 1, regime_bins)
    test_bias = physics_bias[None, :] + args.regime_prior_strength * regime_table[test_regimes]
    scores = predict_scores(model, test_seq, test_cf, device, args.batch) + test_bias
    return test_cands, scores, pre_losses[-1], fine_losses[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="TCN+GRU pretrained candidate selector.")
    parser.add_argument("--root", type=Path, default=Path("step47_physics_ladder/data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/tcn_gru_selector"))
    parser.add_argument("--models", type=str, nargs="*", default=["attn_gru"], choices=["gru", "attn_gru", "lidar_gru", "bigru", "lstm", "tcn", "physics_tcn"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--pre-epochs", type=int, default=14)
    parser.add_argument("--fine-epochs", type=int, default=10)
    parser.add_argument("--epoch-plus", type=int, default=10, help="Full-fit epochs use mean OOF best epoch plus this value.")
    parser.add_argument("--final-repeat", type=int, default=1, help="How many times to include the final measured transition in fine-tuning.")
    parser.add_argument("--include-late-fine", action="store_true", help="Also include late internal transitions during fine-tuning.")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--fold-limit", type=int, default=5)
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--hier-family-gate", action="store_true", help="Use explicit family gate + within-family candidate logits.")
    parser.add_argument("--family-gate-scale", type=float, default=1.0)
    parser.add_argument("--latent-env-experts", type=int, default=1, help="Mixture-of-experts selector count for latent observation/environment regimes.")
    parser.add_argument("--latent-physics-adapter", action="store_true", help="Force hidden state to score candidates through a bounded latent physics coefficient adapter.")
    parser.add_argument("--latent-physics-coeff-scale", type=float, default=0.85)
    parser.add_argument("--fine-ensemble", action="store_true", help="Fork several fine-tune branches from the same pretrained selector and average their scores.")
    parser.add_argument("--fine-branch-scales", type=float, nargs="*", default=[0.65, 1.0, 1.45])
    parser.add_argument("--fine-branch-weight-powers", type=float, nargs="*", default=[0.75, 1.0, 1.35])
    parser.add_argument("--include-pre-branch", action="store_true")
    parser.add_argument("--pairwise-loss-weight", type=float, default=0.0, help="Extra ranking pressure so the best soft-label candidate beats plausible wrong candidates.")
    parser.add_argument("--pairwise-margin", type=float, default=0.10)
    parser.add_argument("--pairwise-min-label-gap", type=float, default=0.04)
    parser.add_argument("--fine-distill-weight", type=float, default=0.0, help="During fine-tuning, keep the pretrained selector distribution as a teacher to reduce final-label noise overfit.")
    parser.add_argument("--fine-distill-temp", type=float, default=0.07)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--fine-lr-scale", type=float, default=0.12)
    parser.add_argument("--freeze-fine-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--prior-strength", type=float, default=0.65)
    parser.add_argument("--regime-prior-strength", type=float, default=0.45)
    parser.add_argument("--reverse-pretrain", action="store_true")
    parser.add_argument("--reverse-pretrain-weight", type=float, default=0.85)
    parser.add_argument("--interp-phases", type=float, nargs="*", default=[])
    parser.add_argument("--interp-weight", type=float, default=0.65)
    parser.add_argument("--interp-min-confidence", type=float, default=0.42)
    parser.add_argument("--interp-forbid-acc-shortcuts", action="store_true")
    parser.add_argument("--mix-copies", type=int, default=0)
    parser.add_argument("--mix-weight", type=float, default=0.45)
    parser.add_argument("--norm-real-only", action="store_true")
    parser.add_argument("--augment-copies", type=int, default=0)
    parser.add_argument("--augment-speed-jitter", type=float, default=0.08)
    parser.add_argument("--augment-accel-noise", type=float, default=0.08)
    parser.add_argument("--augment-weight", type=float, default=0.35)
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load pre-compiled .npy files
    train_x = np.load(root / "train_x.npy")
    train_y = np.load(root / "train_y.npy")
    test_x = np.load(root / "test_x.npy")
    
    with open(root / "train_ids.json", "r") as f:
        train_ids = json.load(f)
    with open(root / "test_ids.json", "r") as f:
        test_ids = json.load(f)

    base_cands = make_candidates(train_x, train_x.shape[1] - 1, horizon=2)
    cand_metrics = {spec.name: metrics(base_cands[:, i], train_y) for i, spec in enumerate(CANDIDATES)}
    oracle_metrics = metrics(base_cands[np.arange(len(train_y)), best_candidate_labels(base_cands, train_y)], train_y)

    active_models = list(dict.fromkeys(args.models))
    oof_scores = {name: np.zeros((len(train_y), len(CANDIDATES)), dtype=np.float32) for name in active_models}
    oof_prior = {name: np.zeros((len(train_y), len(CANDIDATES)), dtype=np.float32) for name in active_models}
    oof_cands = np.zeros((len(train_y), len(CANDIDATES), 3), dtype=np.float32)
    fold_rows = []
    for fold in range(min(args.folds, args.fold_limit)):
        row = {"fold": fold + 1}
        for name in active_models:
            va, val_cands, scores, prior_scores, pre_loss, fine_loss, oracle, train_meta = run_fold(name, train_x, train_y, test_x, train_ids, fold, args.folds, args, device)
            oof_scores[name][va] = scores
            oof_prior[name][va] = prior_scores
            oof_cands[va] = val_cands
            pred_argmax = val_cands[np.arange(len(val_cands)), np.argmax(scores, axis=1)]
            row[f"{name}_argmax"] = metrics(pred_argmax, train_y[va])
            row[f"{name}_pre_loss_last"] = float(pre_loss)
            row[f"{name}_fine_loss_last"] = float(fine_loss)
            row[f"{name}_pre_best_epoch"] = int(train_meta["pre_best_epoch"])
            row[f"{name}_pre_best_hit"] = float(train_meta["pre_best_hit"])
            row[f"{name}_fine_best_epoch"] = int(train_meta["fine_best_epoch"])
            row[f"{name}_fine_best_hit"] = float(train_meta["fine_best_hit"])
            row["oracle"] = oracle
        ens_scores = np.mean(np.stack([oof_scores[name][va] for name in active_models], axis=0), axis=0)
        pred_argmax = val_cands[np.arange(len(val_cands)), np.argmax(ens_scores, axis=1)]
        row["ensemble_argmax"] = metrics(pred_argmax, train_y[va])
        fold_rows.append(row)

    covered = np.abs(oof_cands).sum(axis=(1, 2)) > 0
    eval_y = train_y[covered]
    eval_cands = oof_cands[covered]
    model_eval = {}
    for name in active_models:
        model_eval[name] = {
            "soft": search_temperature(eval_cands, oof_scores[name][covered], eval_y),
            "projected": search_physics_projection(eval_cands, oof_scores[name][covered], oof_prior[name][covered], eval_y),
            "argmax_soft_gate": search_argmax_soft_gate(eval_cands, oof_scores[name][covered], eval_y),
        }
    ens_oof_scores = np.mean(np.stack([oof_scores[name] for name in active_models], axis=0), axis=0)
    ens_prior_scores = np.mean(np.stack([oof_prior[name] for name in active_models], axis=0), axis=0)
    temp_ens = search_temperature(eval_cands, ens_oof_scores[covered], eval_y)
    proj_ens = search_physics_projection(eval_cands, ens_oof_scores[covered], ens_prior_scores[covered], eval_y)
    gate_ens = search_argmax_soft_gate(eval_cands, ens_oof_scores[covered], eval_y)
    np.savez_compressed(
        out_dir / "oof_selector_scores.npz",
        covered=covered,
        y=train_y,
        cands=oof_cands,
        **{f"{name}_scores": oof_scores[name] for name in active_models},
        ens_scores=ens_oof_scores,
        **{f"{name}_prior": oof_prior[name] for name in active_models},
        ens_prior=ens_prior_scores,
        candidate_names=np.asarray([c.name for c in CANDIDATES], dtype=object),
    )

    full_losses = {}
    files = []
    if not args.skip_full:
        pre_best = [int(row.get(f"{name}_pre_best_epoch", args.pre_epochs)) for row in fold_rows for name in active_models]
        fine_best = [int(row.get(f"{name}_fine_best_epoch", args.fine_epochs)) for row in fold_rows for name in active_models if int(row.get(f"{name}_fine_best_epoch", 0)) > 0]
        full_pre_epochs = max(1, int(round(float(np.mean(pre_best)))) + args.epoch_plus) if pre_best else args.pre_epochs
        full_fine_epochs = max(1, int(round(float(np.mean(fine_best)))) + args.epoch_plus) if fine_best else args.fine_epochs
        full_scores = {}
        test_cands = None
        for name in active_models:
            cands_i, scores_i, pre_loss_i, fine_loss_i = train_full_predict(
                name,
                train_x,
                train_y,
                test_x,
                args,
                device,
                pre_epochs=full_pre_epochs,
                fine_epochs=full_fine_epochs,
            )
            test_cands = cands_i
            full_scores[name] = scores_i
            temp_i = float(model_eval[name]["soft"]["temperature"])
            pred_i = soft_select(cands_i, scores_i, temp_i)
            file_i = f"submission_{name}_selector_soft.csv"
            write_submission(out_dir / file_i, test_ids, pred_i)
            files.append(file_i)
            full_losses[name] = {
                "pretrain": float(pre_loss_i),
                "finetune": float(fine_loss_i),
                "full_pre_epochs": int(full_pre_epochs),
                "full_fine_epochs": int(full_fine_epochs),
            }
        assert test_cands is not None
        test_scores_ens = np.mean(np.stack([full_scores[name] for name in active_models], axis=0), axis=0)
        pred_ens = soft_select(test_cands, test_scores_ens, float(temp_ens["temperature"]))
        pred_argmax = test_cands[np.arange(len(test_cands)), np.argmax(test_scores_ens, axis=1)]
        write_submission(out_dir / "submission_selector_ensemble_soft.csv", test_ids, pred_ens)
        write_submission(out_dir / "submission_selector_ensemble_argmax.csv", test_ids, pred_argmax)
        files.extend(["submission_selector_ensemble_soft.csv", "submission_selector_ensemble_argmax.csv"])
        np.savez_compressed(
            out_dir / "test_selector_scores.npz",
            cands=test_cands,
            **{f"{name}_scores": full_scores[name] for name in active_models},
            ens_scores=test_scores_ens,
            candidate_names=np.asarray([c.name for c in CANDIDATES], dtype=object),
        )

    report = {
        "root": str(root),
        "out_dir": str(out_dir),
        "device": str(device),
        "candidate_names": [c.name for c in CANDIDATES],
        "family_names": FAMILY_NAMES,
        "candidate_family": {c.name: FAMILY_NAMES[int(fid)] for c, fid in zip(CANDIDATES, CANDIDATE_FAMILY)},
        "hier_family_gate": bool(args.hier_family_gate),
        "family_gate_scale": float(args.family_gate_scale),
        "latent_env_experts": int(args.latent_env_experts),
        "latent_physics_adapter": bool(args.latent_physics_adapter),
        "latent_physics_coeff_scale": float(args.latent_physics_coeff_scale),
        "final_repeat": int(args.final_repeat),
        "include_late_fine": bool(args.include_late_fine),
        "fine_ensemble": bool(args.fine_ensemble),
        "fine_branch_scales": [float(x) for x in args.fine_branch_scales],
        "fine_branch_weight_powers": [float(x) for x in args.fine_branch_weight_powers],
        "include_pre_branch": bool(args.include_pre_branch),
        "candidate_metrics": cand_metrics,
        "candidate_oracle_metrics": oracle_metrics,
        "fold_rows": fold_rows,
        "model_oof": model_eval,
        "oof_tcn_gru_ensemble_soft": temp_ens,
        "oof_tcn_gru_ensemble_projected": proj_ens,
        "oof_tcn_gru_ensemble_argmax_soft_gate": gate_ens,
        "oof_tcn_gru_ensemble_argmax": metrics(eval_cands[np.arange(len(eval_y)), np.argmax(ens_oof_scores[covered], axis=1)], eval_y),
        "covered_rows": int(np.sum(covered)),
        "full_losses": full_losses,
        "files": files,
    }
    (out_dir / "tcn_gru_selector_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
