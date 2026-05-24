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

# Core 27 from PB 0.6822 (The most stable set)
CANDIDATES = [
    CandidateSpec("p0_d1", 1.00, 0.00, 0.00),
    CandidateSpec("acc_d1_040", 1.00, 0.40, 0.40),
    CandidateSpec("acc_d1_050", 1.00, 0.50, 0.50),
    CandidateSpec("acc_d1_056", 0.99, 0.56, 0.56),
    CandidateSpec("acc_d1_060", 1.00, 0.60, 0.60),
    CandidateSpec("frenet_best", 0.99, 0.96, -0.08),
    CandidateSpec("frenet_par090_perp000", 0.99, 0.90, 0.00),
    CandidateSpec("frenet_par100_perp000", 0.99, 1.00, 0.00),
    CandidateSpec("frenet_par100_perp_neg010", 1.00, 1.00, -0.10),
    CandidateSpec("frenet_par090_perp020", 0.98, 0.90, 0.20),
    CandidateSpec("frenet_par080_perp020", 1.01, 0.80, 0.20),
    CandidateSpec("frenet_par110_perp_neg020", 0.97, 1.10, -0.20),
    CandidateSpec("frenet_fast_par100", 1.03, 1.00, -0.08),
    CandidateSpec("frenet_slow_par100", 0.95, 1.00, -0.08),
    CandidateSpec("jerk_small_pos", 0.99, 0.80, -0.05, jerk=0.08),
    CandidateSpec("jerk_small_neg", 0.99, 0.80, -0.05, jerk=-0.08),
    CandidateSpec("frenet_par070_perp_neg020", 0.99, 0.70, -0.20),
    CandidateSpec("frenet_par120_perp_neg020", 0.99, 1.20, -0.20),
    CandidateSpec("frenet_par120_perp020", 0.99, 1.20, 0.20),
    CandidateSpec("frenet_fast_par120_perp_neg020", 1.04, 1.20, -0.20),
    CandidateSpec("frenet_slow_par070_perp020", 0.93, 0.70, 0.20),
    CandidateSpec("latency_short_frenet_best_085", 0.99, 0.96, -0.08, time_scale=0.85),
    CandidateSpec("latency_short_frenet_best_092", 0.99, 0.96, -0.08, time_scale=0.92),
    CandidateSpec("latency_long_frenet_best_108", 0.99, 0.96, -0.08, time_scale=1.08),
    CandidateSpec("latency_long_frenet_best_115", 0.99, 0.96, -0.08, time_scale=1.15),
    CandidateSpec("latency_long_turn_neg_110", 0.99, 1.10, -0.20, time_scale=1.10),
    CandidateSpec("latency_short_turn_pos_090", 0.98, 0.90, 0.20, time_scale=0.90),
]

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
    acc_perp = acc - acc_par
    
    # Correct Physics Scaling for t=2 steps (80ms)
    spec_v_scale = horizon
    spec_acc_scale = 0.5 * (horizon**2)
    spec_jerk_scale = (1.0/6.0) * (horizon**3)
    
    preds = []
    for spec in CANDIDATES:
        ts = spec.time_scale
        ts2 = ts ** 2
        ts3 = ts ** 3
        
        cand = (p0 
                + spec.d1 * (spec_v_scale * ts) * d1 
                + spec.d2 * (spec_v_scale * ts) * d2 
                + spec.par * (spec_acc_scale * ts2) * acc_par 
                + spec.perp * (spec_acc_scale * ts2) * acc_perp 
                + spec.jerk * (spec_jerk_scale * ts3) * jerk)
        preds.append(cand)
        
    res = np.stack(preds, axis=1).astype(np.float32)
    if not is_batch:
        res = res[0]
    return res
