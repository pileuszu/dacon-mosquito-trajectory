import numpy as np
from dataclasses import dataclass
from pathlib import Path

EPS = 1e-8

@dataclass(frozen=True)
class CandidateSpec:
    name: str
    d1: float
    par: float
    perp: float
    ts: float
    jerk: float = 0.0

def get_candidate_specs():
    specs = []
    # Base Pruned Grid (N=560)
    d1_vals = [0.0, 0.8, 1.0, 1.2]
    par_vals = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    perp_vals = [-1.5, -0.5, 0.0, 0.5, 1.5]
    ts_vals = [0.8, 1.0, 1.2, 1.4]
    
    for d in d1_vals:
        for p in par_vals:
            for n in perp_vals:
                for ts in ts_vals:
                    specs.append(CandidateSpec(f"grid_d{d}_p{p}_n{n}_ts{ts}", d, p, n, ts))
    
    specs.append(CandidateSpec("jerk_pos", 1.0, 1.0, 0.0, 1.0, jerk=0.5))
    specs.append(CandidateSpec("jerk_neg", 1.0, 1.0, 0.0, 1.0, jerk=-0.5))
    
    return specs

CANDIDATES_GLOBAL = get_candidate_specs()

def make_candidates(x, priors=None, end_idx=-1, horizon=2):
    """
    x: (B, T, 3) or (T, 3)
    priors: list of (B, 3) or (3,) regression predictions
    """
    is_batch = x.ndim == 3
    if not is_batch:
        x = x[np.newaxis, ...]
        if priors is not None:
            priors = [p[np.newaxis, ...] if p is not None else None for p in priors]
            
    batch_size = x.shape[0]
    p0 = x[:, end_idx]
    d1 = x[:, end_idx] - x[:, end_idx - 1]
    d2 = x[:, end_idx - 1] - x[:, end_idx - 2]
    d3 = x[:, end_idx - 2] - x[:, end_idx - 3]
    
    acc = d1 - d2
    prev_acc = d2 - d3
    
    speed = np.linalg.norm(d1, axis=1, keepdims=True)
    tangent = d1 / (speed + EPS)
    acc_par_scalar = np.sum(acc * tangent, axis=1, keepdims=True)
    acc_par = acc_par_scalar * tangent
    acc_perp_vec = acc - acc_par
    
    v_scale = horizon
    acc_scale = 0.5 * (horizon**2)
    jerk_scale = (1.0/6.0) * (horizon**3)
    
    preds = []
    
    # 1. Global Physical Candidates
    for spec in CANDIDATES_GLOBAL:
        ts = spec.ts
        ts2 = ts ** 2
        ts3 = ts ** 3
        cand = (p0 
                + spec.d1 * (v_scale * ts) * d1 
                + spec.par * (acc_scale * ts2) * acc_par 
                + spec.perp * (acc_scale * ts2) * acc_perp_vec
                + spec.jerk * (jerk_scale * ts3) * (acc - prev_acc))
        preds.append(cand)
        
    # 2. Local Refinement Grid around Priors
    # Offsets in cm: [-0.6, -0.3, 0.0, 0.3, 0.6] -> 5x5x5 is too many.
    # We use a focused cross + corners (27 points)
    offsets = [-0.006, 0.0, 0.006] # 0.6cm offsets
    if priors:
        for p_idx, p_pos in enumerate(priors):
            if p_pos is None: continue
            for dx in offsets:
                for dy in offsets:
                    for dz in offsets:
                        # Only add if not [0,0,0] (already have base prior)
                        # Actually just add all 27 to be safe
                        local_cand = p_pos + np.array([dx, dy, dz], dtype=np.float32)
                        preds.append(local_cand)
        
    res = np.stack(preds, axis=1).astype(np.float32)
    if not is_batch:
        res = res[0]
    return res
