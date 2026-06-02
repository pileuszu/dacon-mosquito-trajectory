from __future__ import annotations

import argparse
import json
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import step47_physics_ladder.train_tcn_gru_candidate_selector as base


def local_frame(x: np.ndarray, end_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, d1, acc = base.motion_terms(x, end_idx)
    t = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + base.EPS)
    acc_perp = acc - np.sum(acc * t, axis=1, keepdims=True) * t
    n_norm = np.linalg.norm(acc_perp, axis=1, keepdims=True)
    n = acc_perp / (n_norm + base.EPS)
    fallback = np.zeros_like(n)
    axis = np.argmin(np.abs(t), axis=1)
    fallback[np.arange(len(x)), axis] = 1.0
    fallback = fallback - np.sum(fallback * t, axis=1, keepdims=True) * t
    fallback /= np.linalg.norm(fallback, axis=1, keepdims=True) + base.EPS
    n = np.where(n_norm > 1e-6, n, fallback)
    b = np.cross(t, n)
    b /= np.linalg.norm(b, axis=1, keepdims=True) + base.EPS
    speed = np.linalg.norm(d1, axis=1, keepdims=True)
    return t.astype(np.float32), n.astype(np.float32), b.astype(np.float32), speed.astype(np.float32)


def vector_to_local(vec: np.ndarray, basis: tuple[np.ndarray, np.ndarray, np.ndarray], scale: np.ndarray) -> np.ndarray:
    t, n, b = basis
    comp = np.stack(
        [
            np.sum(vec * t[:, None, :], axis=2),
            np.sum(vec * n[:, None, :], axis=2),
            np.sum(vec * b[:, None, :], axis=2),
        ],
        axis=2,
    )
    return (comp / (scale[:, None, :] + base.EPS)).astype(np.float32)


def local_to_vector(local: np.ndarray, basis: tuple[np.ndarray, np.ndarray, np.ndarray], scale: np.ndarray) -> np.ndarray:
    t, n, b = basis
    return (local[:, :, 0:1] * t[:, None, :] + local[:, :, 1:2] * n[:, None, :] + local[:, :, 2:3] * b[:, None, :]) * scale[:, None, :]


def cap_vectors(vec: np.ndarray, cap: float) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=2, keepdims=True)
    factor = np.minimum(1.0, cap / (norm + base.EPS))
    return vec * factor


def family_targets(candidates: np.ndarray, target: np.ndarray) -> np.ndarray:
    best = base.best_candidate_labels(candidates, target)
    names = np.asarray([c.name for c in base.CANDIDATES], dtype=object)
    out = np.zeros(len(best), dtype=np.int64)
    for i, name in enumerate(names[best].astype(str)):
        if name.startswith("latency"):
            out[i] = 3
        elif "jerk" in name:
            out[i] = 2
        elif name.startswith("frenet"):
            out[i] = 1
        else:
            out[i] = 0
    return out


def make_rows(
    x: np.ndarray,
    target: np.ndarray,
    end_idx: int,
    horizon: int,
    *,
    cap: float,
    low: float,
    high: float,
    far_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cands = base.make_candidates(x, end_idx, horizon=horizon)
    cf = base.make_candidate_features(x, end_idx, cands, horizon=horizon)
    t, n, b, speed = local_frame(x, end_idx)
    scale = np.maximum(speed * float(horizon), base.EPS)
    residual = target[:, None, :] - cands
    residual = cap_vectors(residual, cap)
    local = vector_to_local(residual, (t, n, b), scale)
    err = np.linalg.norm(target[:, None, :] - cands, axis=2)
    boundary = (err >= low) & (err <= high)
    easy = err < low
    weights = np.where(boundary, 1.0, np.where(easy, 0.20, far_weight)).astype(np.float32)
    hit_after = (np.linalg.norm(target[:, None, :] - (cands + residual), axis=2) <= base.R_HIT).astype(np.float32)
    fam = family_targets(cands, target)
    return cf, local, weights, cands, hit_after, fam


def build_pretrain(
    x: np.ndarray,
    *,
    cap: float,
    low: float,
    high: float,
    far_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feats, targets, weights, families = [], [], [], []
    for horizon in (1, 2):
        min_end = 8
        max_end = x.shape[1] - horizon - 1
        for end_idx in range(min_end, max_end + 1):
            cf, local, w, _, _, fam = make_rows(
                x,
                x[:, end_idx + horizon],
                end_idx,
                horizon,
                cap=cap,
                low=low,
                high=high,
                far_weight=far_weight,
            )
            feats.append(cf.reshape(-1, cf.shape[-1]))
            targets.append(local.reshape(-1, 3))
            weights.append(w.reshape(-1))
            families.append(np.repeat(fam, len(base.CANDIDATES)))
    return np.vstack(feats), np.vstack(targets), np.concatenate(weights), np.concatenate(families)


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float = 0.04):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TinyCorrectionNet(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(0.04),
        )
        self.blocks = nn.Sequential(
            ResidualMLPBlock(hidden),
            ResidualMLPBlock(hidden),
        )
        self.delta = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 3),
        )
        self.env = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 4),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.blocks(self.stem(x))
        return self.delta(h), self.env(h)


def train_net(
    model: TinyCorrectionNet,
    cf: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    family: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    *,
    stage: str,
    val_payload: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> None:
    ds = TensorDataset(torch.from_numpy(cf), torch.from_numpy(target), torch.from_numpy(weight), torch.from_numpy(family))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr if stage == "pretrain" else args.lr * args.fine_lr_scale, weight_decay=1e-4)
    best_state = base.clone_state_dict(model)
    best_hit = -1.0
    if val_payload is not None:
        current = evaluate(model, *val_payload, args=args, device=device)
        best_hit = float(current["gate"]["metrics"]["hit"])
    wait = 0
    for epoch in range(1, args.epochs + 1 if stage == "pretrain" else args.fine_epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, yb, wb, fb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            fb = fb.to(device)
            opt.zero_grad(set_to_none=True)
            pred, env = model(xb)
            reg = ((pred - yb) ** 2).sum(dim=1)
            env_loss = nn.functional.cross_entropy(env, fb, reduction="none")
            loss = ((reg + args.env_loss_weight * env_loss) * wb).sum() / (wb.sum() + 1e-8)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            n += len(xb)
        if val_payload is not None:
            m = evaluate(model, *val_payload, args=args, device=device)
            hit = float(m["gate"]["metrics"]["hit"])
            if hit > best_hit:
                best_hit = hit
                best_state = base.clone_state_dict(model)
                wait = 0
            else:
                wait += 1
            print(
                "GAUGE",
                f"stage={stage}",
                f"epoch={epoch}",
                f"loss={total / max(n, 1):.6f}",
                f"soft_hit={m['soft']['metrics']['hit']:.4f}",
                f"gate_hit={m['gate']['metrics']['hit']:.4f}",
                f"best_gate={best_hit:.4f}",
                f"wait={wait}/{args.patience}",
                flush=True,
            )
            if epoch >= args.min_epochs and wait >= args.patience:
                break
        else:
            print(
                "FULL_GAUGE",
                f"stage={stage}",
                f"epoch={epoch}",
                f"loss={total / max(n, 1):.6f}",
                flush=True,
            )
    if val_payload is not None:
        base.load_state_dict_cpu(model, best_state)


@torch.no_grad()
def predict_delta(model: TinyCorrectionNet, cf: np.ndarray, args: argparse.Namespace, device: torch.device) -> np.ndarray:
    model.eval()
    outs = []
    for start in range(0, len(cf), args.batch):
        xb = torch.from_numpy(cf[start : start + args.batch]).to(device)
        pred, _ = model(xb)
        outs.append(pred.detach().cpu().numpy())
    return np.vstack(outs)


def evaluate(
    model: TinyCorrectionNet,
    cf: np.ndarray,
    cands: np.ndarray,
    true: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
    scale: np.ndarray,
    scores: np.ndarray,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    flat = cf.reshape(-1, cf.shape[-1])
    delta = predict_delta(model, flat, args, device).reshape(cands.shape[0], cands.shape[1], 3)
    delta_vec = local_to_vector(delta, basis, scale)
    delta_vec = cap_vectors(delta_vec, args.cap)
    corrected = cands + args.apply_scale * delta_vec
    return {
        "soft": base.search_temperature(corrected, scores, true),
        "gate": base.search_argmax_soft_gate(corrected, scores, true),
        "argmax": base.metrics(corrected[np.arange(len(true)), np.argmax(scores, axis=1)], true),
    }


def predict_corrected_candidates(
    model: TinyCorrectionNet,
    cf: np.ndarray,
    cands: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
    scale: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    flat = cf.reshape(-1, cf.shape[-1])
    delta = predict_delta(model, flat, args, device).reshape(cands.shape[0], cands.shape[1], 3)
    delta_vec = local_to_vector(delta, basis, scale)
    delta_vec = cap_vectors(delta_vec, args.cap)
    return cands + args.apply_scale * delta_vec


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny local-frame correction for 1cm boundary rows.")
    parser.add_argument("--root", type=Path, default=Path("step47_physics_ladder/data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/boundary_tiny_correction_fold1"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--fine-epochs", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--fine-lr-scale", type=float, default=0.18)
    parser.add_argument("--cap", type=float, default=0.006)
    parser.add_argument("--apply-scale", type=float, default=0.75)
    parser.add_argument("--low", type=float, default=0.007)
    parser.add_argument("--high", type=float, default=0.017)
    parser.add_argument("--far-weight", type=float, default=0.04)
    parser.add_argument("--prior-strength", type=float, default=0.65)
    parser.add_argument("--regime-prior-strength", type=float, default=0.45)
    parser.add_argument("--score-bank", type=Path, default=None, help="OOF selector score bank from train_tcn_gru_candidate_selector.py")
    parser.add_argument("--score-key", type=str, default="ens_scores")
    parser.add_argument("--make-test", action="store_true")
    parser.add_argument("--test-score-bank", type=Path, default=None, help="Full-fit selector test score bank from train_tcn_gru_candidate_selector.py")
    parser.add_argument("--test-score-key", type=str, default="ens_scores")
    parser.add_argument("--save-val-pred", action="store_true")
    parser.add_argument("--env-loss-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    root = args.root.resolve()
    
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    base.set_torch_seed(args.seed)

    # Load pre-compiled .npy files
    train_x = np.load(root / "train_x.npy")
    train_y = np.load(root / "train_y.npy")
    
    with open(root / "train_ids.json", "r") as f:
        ids = json.load(f)

    fold_ids = np.asarray([base.stable_fold_id(sample_id, args.folds) for sample_id in ids])
    va = fold_ids == args.fold
    tr = ~va

    pre_cf, pre_target, pre_weight, pre_family = build_pretrain(
        train_x[tr],
        cap=args.cap,
        low=args.low,
        high=args.high,
        far_weight=args.far_weight,
    )
    final_cf3, final_local3, final_w2, train_cands, _, final_family = make_rows(
        train_x[tr],
        train_y[tr],
        train_x.shape[1] - 1,
        2,
        cap=args.cap,
        low=args.low,
        high=args.high,
        far_weight=args.far_weight,
    )
    fine_cf = final_cf3.reshape(-1, final_cf3.shape[-1])
    fine_target = final_local3.reshape(-1, 3)
    fine_weight = (final_w2.reshape(-1) * 1.8).astype(np.float32)
    fine_family = np.repeat(final_family, len(base.CANDIDATES))

    _, _, cm, cs = base.normalize_fit(np.zeros((1, 6, len(base.SEQ_FEATURE_NAMES)), dtype=np.float32), final_cf3)
    pre_cf = ((pre_cf - cm) / cs).astype(np.float32)
    fine_cf = ((fine_cf - cm) / cs).astype(np.float32)

    val_cands = base.make_candidates(train_x[va], train_x.shape[1] - 1, horizon=2)
    val_cf3 = base.make_candidate_features(train_x[va], train_x.shape[1] - 1, val_cands, horizon=2)
    val_cf3 = ((val_cf3 - cm) / cs).astype(np.float32)
    t, n, b, speed = local_frame(train_x[va], train_x.shape[1] - 1)
    val_scale = np.maximum(speed * 2.0, base.EPS)

    physics_bias = base.candidate_physics_bias(train_cands, train_y[tr]) * args.prior_strength
    bins = base.fit_regime_bins(train_x[tr], train_x.shape[1] - 1)
    train_regimes = base.assign_regimes(train_x[tr], train_x.shape[1] - 1, bins)
    val_regimes = base.assign_regimes(train_x[va], train_x.shape[1] - 1, bins)
    regime_table = base.candidate_regime_bias(train_cands, train_y[tr], train_regimes, regime_count=18)
    val_scores = physics_bias[None, :] + args.regime_prior_strength * regime_table[val_regimes]
    score_source = "physics_regime_prior"
    if args.score_bank is not None:
        z = np.load(args.score_bank, allow_pickle=True)
        bank_cands = z["cands"]
        bank_scores = z[args.score_key]
        bank_names = [str(x) for x in z["candidate_names"].tolist()]
        local_names = [c.name for c in base.CANDIDATES]
        if bank_scores.shape[:2] != (len(train_y), len(base.CANDIDATES)):
            raise ValueError(f"score bank shape mismatch: {bank_scores.shape}")
        if bank_names != local_names:
            raise ValueError(f"candidate name mismatch:\nbank={bank_names}\nlocal={local_names}")
        max_cand_delta = float(np.max(np.abs(bank_cands[va] - val_cands)))
        if max_cand_delta > 1e-5:
            raise ValueError(f"score bank candidate mismatch on validation fold: max delta {max_cand_delta}")
        val_scores = bank_scores[va].astype(np.float32)
        score_source = f"{args.score_bank}:{args.score_key}"
    val_payload = (val_cf3, val_cands, train_y[va], (t, n, b), val_scale, val_scores)

    model = TinyCorrectionNet(pre_cf.shape[-1], args.hidden).to(device)
    print("BASELINE", json.dumps({
        "score_source": score_source,
        "soft": base.search_temperature(val_cands, val_scores, train_y[va]),
        "gate": base.search_argmax_soft_gate(val_cands, val_scores, train_y[va]),
    }), flush=True)
    print("TRAIN_ROWS", f"pre={len(pre_cf)}", f"fine={len(fine_cf)}", f"boundary_pre_weight={float(pre_weight.sum()):.1f}", flush=True)
    train_net(model, pre_cf, pre_target, pre_weight, pre_family, args, device, stage="pretrain", val_payload=val_payload)
    train_net(model, fine_cf, fine_target, fine_weight, fine_family, args, device, stage="finetune", val_payload=val_payload)
    result = evaluate(model, *val_payload, args=args, device=device)
    if args.save_val_pred:
        corrected_val = predict_corrected_candidates(model, val_cf3, val_cands, (t, n, b), val_scale, args, device)
        soft_temp = float(result["soft"].get("temperature", 0.07))
        val_soft = base.soft_select(corrected_val, val_scores, soft_temp)
        val_argmax = corrected_val[np.arange(len(corrected_val)), np.argmax(val_scores, axis=1)]
        np.savez_compressed(
            out_dir / "boundary_val_predictions.npz",
            fold=np.asarray([args.fold], dtype=np.int64),
            val_mask=va,
            val_ids=np.asarray(ids, dtype=object)[va],
            y=train_y[va],
            soft=val_soft.astype(np.float32),
            argmax=val_argmax.astype(np.float32),
            soft_temperature=np.asarray([soft_temp], dtype=np.float32),
        )
    result["candidate_oracle"] = base.metrics(
        val_cands[np.arange(np.sum(va)), base.best_candidate_labels(val_cands, train_y[va])],
        train_y[va],
    )
    
    # Save model state dict
    torch.save(model.state_dict(), out_dir / f"boundary_model_fold{args.fold}.pt")
    
    test_files = []
    if args.make_test:
        with open(root / "test_ids.json", "r") as f:
            test_ids = json.load(f)
        test_x = np.load(root / "test_x.npy")

        all_pre_cf, all_pre_target, all_pre_weight, all_pre_family = build_pretrain(
            train_x,
            cap=args.cap,
            low=args.low,
            high=args.high,
            far_weight=args.far_weight,
        )
        all_final_cf3, all_final_local3, all_final_w2, all_train_cands, _, all_final_family = make_rows(
            train_x,
            train_y,
            train_x.shape[1] - 1,
            2,
            cap=args.cap,
            low=args.low,
            high=args.high,
            far_weight=args.far_weight,
        )
        all_fine_cf = all_final_cf3.reshape(-1, all_final_cf3.shape[-1])
        all_fine_target = all_final_local3.reshape(-1, 3)
        all_fine_weight = (all_final_w2.reshape(-1) * 1.8).astype(np.float32)
        all_fine_family = np.repeat(all_final_family, len(base.CANDIDATES))
        _, _, all_cm, all_cs = base.normalize_fit(
            np.zeros((1, 6, len(base.SEQ_FEATURE_NAMES)), dtype=np.float32),
            all_final_cf3,
        )
        all_pre_cf = ((all_pre_cf - all_cm) / all_cs).astype(np.float32)
        all_fine_cf = ((all_fine_cf - all_cm) / all_cs).astype(np.float32)

        full_model = TinyCorrectionNet(all_pre_cf.shape[-1], args.hidden).to(device)
        train_net(full_model, all_pre_cf, all_pre_target, all_pre_weight, all_pre_family, args, device, stage="pretrain", val_payload=None)
        train_net(full_model, all_fine_cf, all_fine_target, all_fine_weight, all_fine_family, args, device, stage="finetune", val_payload=None)
        
        torch.save(full_model.state_dict(), out_dir / "boundary_model_full.pt")

        test_cands = base.make_candidates(test_x, test_x.shape[1] - 1, horizon=2)
        test_cf3 = base.make_candidate_features(test_x, test_x.shape[1] - 1, test_cands, horizon=2)
        test_cf3_norm = ((test_cf3 - all_cm) / all_cs).astype(np.float32)
        tt, tn, tb, tspeed = local_frame(test_x, test_x.shape[1] - 1)
        test_scale = np.maximum(tspeed * 2.0, base.EPS)

        all_physics_bias = base.candidate_physics_bias(all_train_cands, train_y) * args.prior_strength
        all_bins = base.fit_regime_bins(train_x, train_x.shape[1] - 1)
        all_train_regimes = base.assign_regimes(train_x, train_x.shape[1] - 1, all_bins)
        test_regimes = base.assign_regimes(test_x, test_x.shape[1] - 1, all_bins)
        all_regime_table = base.candidate_regime_bias(all_train_cands, train_y, all_train_regimes, regime_count=18)
        test_scores = all_physics_bias[None, :] + args.regime_prior_strength * all_regime_table[test_regimes]
        if args.test_score_bank is not None:
            tz = np.load(args.test_score_bank, allow_pickle=True)
            test_scores = tz[args.test_score_key].astype(np.float32)
            bank_names = [str(x) for x in tz["candidate_names"].tolist()]
            local_names = [c.name for c in base.CANDIDATES]
            if bank_names != local_names:
                raise ValueError(f"test score bank candidate mismatch:\nbank={bank_names}\nlocal={local_names}")
            if "cands" in tz:
                max_test_delta = float(np.max(np.abs(tz["cands"] - test_cands)))
                if max_test_delta > 1e-5:
                    raise ValueError(f"test score bank candidate mismatch: max delta {max_test_delta}")

        flat_test = test_cf3_norm.reshape(-1, test_cf3_norm.shape[-1])
        delta = predict_delta(full_model, flat_test, args, device).reshape(test_cands.shape[0], test_cands.shape[1], 3)
        delta_vec = cap_vectors(local_to_vector(delta, (tt, tn, tb), test_scale), args.cap)
        corrected = test_cands + args.apply_scale * delta_vec
        temp = float(result["soft"]["temperature"]) if "temperature" in result["soft"] else 0.03
        pred_soft = base.soft_select(corrected, test_scores, temp)
        pred_argmax = corrected[np.arange(len(corrected)), np.argmax(test_scores, axis=1)]
        soft_file = out_dir / "submission_boundary_tiny_soft.csv"
        arg_file = out_dir / "submission_boundary_tiny_argmax.csv"
        base.write_submission(soft_file, test_ids, pred_soft)
        base.write_submission(arg_file, test_ids, pred_argmax)
        test_files = [str(soft_file), str(arg_file)]
    result["test_files"] = test_files
    (out_dir / "boundary_tiny_correction_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
