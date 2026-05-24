import numpy as np
from dataclasses import dataclass

EPS = 1e-8

@dataclass(frozen=True)
class CandidateSpec:
    name: str
    d1: float
    par: float
    perp: float
    d2: float = 0.0
    jerk: float = 0.0
    time_scale: float = 1.0

def get_candidate_specs():
    specs = []
    
    # 1. Wide Hybrid Grid from Step 9
    par_vals = [-0.5, 0.0, 0.4, 0.7, 1.0, 1.3, 1.6, 2.0] 
    perp_vals = [-1.5, -1.0, -0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 1.0, 1.5]
    ts_vals = [0.7, 0.9, 1.0, 1.1, 1.3]
    
    for p in par_vals:
        for n in perp_vals:
            for ts in ts_vals:
                specs.append(CandidateSpec(f"wide_p{p}_n{n}_ts{ts}", 1.0, p, n, time_scale=ts))
                
    # Add jerk variants from Step 9
    for j in [-0.2, 0.2]:
        specs.append(CandidateSpec(f"jerk_v_{j}", 1.0, 1.0, 0.0, jerk=j))
        
    return specs

CANDIDATES = get_candidate_specs()

def make_candidates(x, end_idx=-1, horizon=2):
    is_batch = x.ndim == 3
    if not is_batch:
        x = x[np.newaxis, ...]
        
    p0 = x[:, end_idx]
    d1 = x[:, end_idx] - x[:, end_idx - 1]
    d2 = x[:, end_idx - 1] - x[:, end_idx - 2]
    d3 = x[:, end_idx - 2] - x[:, end_idx - 3]
    
    acc = d1 - d2
    prev_acc = d2 - d3
    jerk = acc - prev_acc
    
    speed = np.linalg.norm(d1, axis=1, keepdims=True)
    tangent = d1 / (speed + EPS)
    
    acc_par_scalar = np.sum(acc * tangent, axis=1, keepdims=True)
    acc_par = acc_par_scalar * tangent
    acc_perp_vec = acc - acc_par
    
    # Physics Scaling
    v_scale = horizon
    acc_scale = 0.5 * (horizon**2)
    jerk_scale = (1.0/6.0) * (horizon**3)
    
    preds = []
    for spec in CANDIDATES:
        ts = spec.time_scale
        ts2 = ts ** 2
        ts3 = ts ** 3
        
        cand = (p0 
                + spec.d1 * (v_scale * ts) * d1 
                + spec.d2 * (v_scale * ts) * d2 
                + spec.par * (acc_scale * ts2) * acc_par 
                + spec.perp * (acc_scale * ts2) * acc_perp_vec
                + spec.jerk * (jerk_scale * ts3) * jerk)
        preds.append(cand)
        
    res = np.stack(preds, axis=1).astype(np.float32)
    if not is_batch:
        res = res[0]
    return res
