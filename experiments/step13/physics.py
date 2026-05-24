import numpy as np
from dataclasses import dataclass

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
    
    # Add High-Energy "Panic" candidates for sudden turns/stops
    # Extreme lateral acceleration and extreme deceleration
    for ts in [1.0, 1.3]:
        specs.append(CandidateSpec(f"panic_turn_L_ts{ts}", 1.0, 0.0, 3.5, ts))
        specs.append(CandidateSpec(f"panic_turn_R_ts{ts}", 1.0, 0.0, -3.5, ts))
        specs.append(CandidateSpec(f"panic_stop_ts{ts}", 0.5, -2.5, 0.0, ts))
        
    return specs

CANDIDATES_GLOBAL = get_candidate_specs()

def make_candidates(x, priors=None, end_idx=-1, horizon=2):
    """
    Ultimate Physics V2: Adaptive Local Grid and High-Energy Candidates
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
    
    # 1. Global Grid (560 + 6 Panic = 566)
    for spec in CANDIDATES_GLOBAL:
        ts = spec.ts
        ts2 = ts ** 2
        cand = (p0 
                + spec.d1 * (v_scale * ts) * d1 
                + spec.par * (acc_scale * ts2) * acc_par 
                + spec.perp * (acc_scale * ts2) * acc_perp_vec)
        preds.append(cand)
        
    # 2. Velocity-Adaptive Local Refinement (125 points per prior)
    # The search radius grows with speed to handle high-velocity uncertainty
    local_range = 0.012 + 0.3 * speed # (B, 1) - Adaptive range
    
    # Create a unit grid (-1 to 1) for broadcasting
    unit_axis = np.linspace(-1, 1, 5)
    unit_grid = np.stack(np.meshgrid(unit_axis, unit_axis, unit_axis), axis=-1).reshape(-1, 3) # (125, 3)
    
    if priors:
        for p_pos in priors:
            if p_pos is None: continue
            # p_pos: (B, 3)
            # Scale grid per sample based on its speed
            scaled_grid = unit_grid[np.newaxis, :, :] * local_range[:, np.newaxis, :] # (B, 125, 3)
            local_cands = p_pos[:, np.newaxis, :] + scaled_grid # (B, 125, 3)
            
            for j in range(125):
                preds.append(local_cands[:, j, :])
                
        # 3. Midpoint Prior (Strategic Candidate)
        # If Step 7 and Step 12 are different, the truth is often between them
        if len(priors) >= 2 and priors[0] is not None and priors[1] is not None:
            mid_prior = (priors[0] + priors[1]) / 2.0
            preds.append(mid_prior)
            # Add a small cross around midpoint (6 points)
            m_range = 0.005 # Fixed 0.5cm
            for axis in range(3):
                for sign in [-1, 1]:
                    offset = np.zeros((batch_size, 3))
                    offset[:, axis] = sign * m_range
                    preds.append(mid_prior + offset)
        
    res = np.stack(preds, axis=1).astype(np.float32)
    if not is_batch:
        res = res[0]
    return res
