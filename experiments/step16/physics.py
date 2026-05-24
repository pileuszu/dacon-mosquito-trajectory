import numpy as np
from dataclasses import dataclass

EPS = 1e-8

@dataclass(frozen=True)
class CandidateSpec:
    name: str
    par: float
    perp: float
    ts: float
    jerk: float = 0.0

def get_hybrid_specs():
    specs = []
    # 1. Step 9 Fine Grid (8 x 11 x 5 = 440)
    par_vals = [-0.5, 0.0, 0.4, 0.7, 1.0, 1.3, 1.6, 2.0] 
    perp_vals = [-1.5, -1.0, -0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 1.0, 1.5]
    ts_vals = [0.7, 0.9, 1.0, 1.1, 1.3]
    
    for p in par_vals:
        for n in perp_vals:
            for ts in ts_vals:
                specs.append(CandidateSpec(f"fine_p{p}_n{n}_ts{ts}", p, n, ts))
                
    # 2. Add Panic candidates from Step 15
    for ts in [1.0, 1.3]:
        specs.append(CandidateSpec(f"panic_L_ts{ts}", 0.0, 3.5, ts))
        specs.append(CandidateSpec(f"panic_R_ts{ts}", 0.0, -3.5, ts))
        specs.append(CandidateSpec(f"panic_stop_ts{ts}", -2.5, 0.0, ts))
        
    return specs

CANDIDATES_GLOBAL = get_hybrid_specs()

def make_candidates(x, priors=None, end_idx=-1, horizon=2):
    """
    Step 16 Hybrid Physics: Step 9 Fine Grid + Step 15 Adaptive Range
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
    
    acc = d1 - d2
    speed = np.linalg.norm(d1, axis=1, keepdims=True)
    tangent = d1 / (speed + EPS)
    acc_par_scalar = np.sum(acc * tangent, axis=1, keepdims=True)
    acc_par = acc_par_scalar * tangent
    acc_perp_vec = acc - acc_par
    
    v_scale = horizon
    acc_scale = 0.5 * (horizon**2)
    
    preds = []
    
    # 1. Global Fine Grid
    for spec in CANDIDATES_GLOBAL:
        ts = spec.ts
        ts2 = ts ** 2
        # d1_scale is usually 1.0 in these specs
        cand = (p0 
                + (v_scale * ts) * d1 
                + spec.par * (acc_scale * ts2) * acc_par 
                + spec.perp * (acc_scale * ts2) * acc_perp_vec)
        preds.append(cand)
        
    # 2. Adaptive Local Grid (around priors)
    local_range = 0.012 + 0.3 * speed
    unit_axis = np.linspace(-1, 1, 5)
    unit_grid = np.stack(np.meshgrid(unit_axis, unit_axis, unit_axis), axis=-1).reshape(-1, 3) # (125, 3)
    
    if priors:
        for p_pos in priors:
            if p_pos is None: continue
            scaled_grid = unit_grid[np.newaxis, :, :] * local_range[:, np.newaxis, :]
            local_cands = p_pos[:, np.newaxis, :] + scaled_grid
            for j in range(125):
                preds.append(local_cands[:, j, :])
        
        # Midpoint Prior
        if len(priors) >= 2 and priors[0] is not None and priors[1] is not None:
            mid_prior = (priors[0] + priors[1]) / 2.0
            preds.append(mid_prior)

    res = np.stack(preds, axis=1).astype(np.float32)
    if not is_batch:
        res = res[0]
    return res
