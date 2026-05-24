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
    damping: float = 0.0  # Non-dimensional damping parameter (lambda = beta * T)

def get_damping_factors(lam):
    """
    Computes analytical continuous-time damping scaling factors:
    F_v = (1 - e^-L) / L
    F_a = 2 * (L - 1 + e^-L) / L^2
    F_j = 6 * (L^2/2 - L + 1 - e^-L) / L^3
    
    Uses Maclaurin series expansions when L is close to 0 to prevent division by zero.
    """
    eps = 1e-4
    
    # Check if lam is a numpy array
    if isinstance(lam, np.ndarray):
        cond = lam < eps
        # F_v series: 1 - L/2 + L^2/6 - L^3/24
        fv_series = 1.0 - lam/2.0 + (lam**2)/6.0 - (lam**3)/24.0
        fv_exact = (1.0 - np.exp(-np.maximum(lam, eps))) / np.maximum(lam, eps)
        F_v = np.where(cond, fv_series, fv_exact)
        
        # F_a series: 1 - L/3 + L^2/12 - L^3/60
        fa_series = 1.0 - lam/3.0 + (lam**2)/12.0 - (lam**3)/60.0
        fa_exact = 2.0 * (lam - 1.0 + np.exp(-np.maximum(lam, eps))) / (np.maximum(lam, eps)**2)
        F_a = np.where(cond, fa_series, fa_exact)
        
        # F_j series: 1 - L/4 + L^2/20 - L^3/120
        fj_series = 1.0 - lam/4.0 + (lam**2)/20.0 - (lam**3)/120.0
        fj_exact = 6.0 * (0.5*(lam**2) - lam + 1.0 - np.exp(-np.maximum(lam, eps))) / (np.maximum(lam, eps)**3)
        F_j = np.where(cond, fj_series, fj_exact)
    else:
        # Scalar version
        if lam < eps:
            F_v = 1.0 - lam/2.0 + (lam**2)/6.0 - (lam**3)/24.0
            F_a = 1.0 - lam/3.0 + (lam**2)/12.0 - (lam**3)/60.0
            F_j = 1.0 - lam/4.0 + (lam**2)/20.0 - (lam**3)/120.0
        else:
            F_v = (1.0 - np.exp(-lam)) / lam
            F_a = 2.0 * (lam - 1.0 + np.exp(-lam)) / (lam**2)
            F_j = 6.0 * (0.5*(lam**2) - lam + 1.0 - np.exp(-lam)) / (lam**3)
            
    return F_v, F_a, F_j

def get_candidate_specs():
    specs = []
    
    # 7 * 7 * 3 * 4 = 588 candidates
    par_vals = [-0.5, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
    perp_vals = [-1.5, -0.8, -0.3, 0.0, 0.3, 0.8, 1.5]  # Balanced lateral limits
    ts_vals = [0.5, 0.8, 1.0, 1.2]
    damping_vals = [0.0, 0.5, 2.0, 5.0]  # Non-dimensional damping lambda
    
    for p in par_vals:
        for n in perp_vals:
            for ts in ts_vals:
                for dmp in damping_vals:
                    specs.append(CandidateSpec(
                        name=f"dmp_p{p}_n{n}_ts{ts}_d{dmp}",
                        d1=1.0,
                        par=p,
                        perp=n,
                        time_scale=ts,
                        damping=dmp
                    ))
                
    # Add specialized jerk variants (no damping)
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
    
    # Base Physical Taylor scaling
    v_scale = horizon
    acc_scale = 0.5 * (horizon**2)
    jerk_scale = (1.0/6.0) * (horizon**3)
    
    # Vectorized / Batch calculation for candidates
    # Create arrays of damping and spec values
    dmps = np.array([s.damping for s in CANDIDATES], dtype=np.float32)
    F_v, F_a, F_j = get_damping_factors(dmps)
    
    preds = []
    for i, spec in enumerate(CANDIDATES):
        ts = spec.time_scale
        ts2 = ts ** 2
        ts3 = ts ** 3
        
        fv = F_v[i]
        fa = F_a[i]
        fj = F_j[i]
        
        # Apply the analytical continuous-time damping factors
        cand = (p0 
                + spec.d1 * (v_scale * ts * fv) * d1 
                + spec.d2 * (v_scale * ts * fv) * d2 
                + spec.par * (acc_scale * ts2 * fa) * acc_par 
                + spec.perp * (acc_scale * ts2 * fa) * acc_perp_vec
                + spec.jerk * (jerk_scale * ts3 * fj) * jerk)
        preds.append(cand)
        
    res = np.stack(preds, axis=1).astype(np.float32)
    if not is_batch:
        res = res[0]
    return res
