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
    damping: float = 0.0

def get_damping_factors(lam):
    eps = 1e-4
    if isinstance(lam, np.ndarray):
        cond = lam < eps
        fv_series = 1.0 - lam/2.0 + (lam**2)/6.0 - (lam**3)/24.0
        fv_exact = (1.0 - np.exp(-np.maximum(lam, eps))) / np.maximum(lam, eps)
        F_v = np.where(cond, fv_series, fv_exact)
        
        fa_series = 1.0 - lam/3.0 + (lam**2)/12.0 - (lam**3)/60.0
        fa_exact = 2.0 * (lam - 1.0 + np.exp(-np.maximum(lam, eps))) / (np.maximum(lam, eps)**2)
        F_a = np.where(cond, fa_series, fa_exact)
        
        fj_series = 1.0 - lam/4.0 + (lam**2)/20.0 - (lam**3)/120.0
        fj_exact = 6.0 * (0.5*(lam**2) - lam + 1.0 - np.exp(-np.maximum(lam, eps))) / (np.maximum(lam, eps)**3)
        F_j = np.where(cond, fj_series, fj_exact)
    else:
        if lam < eps:
            F_v = 1.0 - lam/2.0 + (lam**2)/6.0 - (lam**3)/24.0
            F_a = 1.0 - lam/3.0 + (lam**2)/12.0 - (lam**3)/60.0
            F_j = 1.0 - lam/4.0 + (lam**2)/20.0 - (lam**3)/120.0
        else:
            F_v = (1.0 - np.exp(-lam)) / lam
            F_a = 2.0 * (lam - 1.0 + np.exp(-lam)) / (lam**2)
            F_j = 6.0 * (0.5*(lam**2) - lam + 1.0 - np.exp(-lam)) / (lam**3)
            
    return F_v, F_a, F_j

def get_slow_specs():
    specs = []
    # 4 * 5 = 20 specs
    par_vals = [-0.2, 0.0, 0.2, 0.5]
    perp_vals = [-0.3, -0.1, 0.0, 0.1, 0.3]
    for p in par_vals:
        for n in perp_vals:
            specs.append(CandidateSpec(
                name=f"slow_p{p}_n{n}",
                d1=1.0,
                par=p,
                perp=n,
                time_scale=1.0,
                damping=0.0
            ))
    return specs

def get_fast_specs():
    specs = []
    # 8 * 11 * 5 * 2 = 880 specs
    par_vals = [-0.5, 0.0, 0.4, 0.7, 1.0, 1.3, 1.6, 2.0]
    perp_vals = [-1.5, -1.0, -0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 1.0, 1.5]
    ts_vals = [0.7, 0.9, 1.0, 1.1, 1.3]
    damping_vals = [0.0, 0.5]
    
    for p in par_vals:
        for n in perp_vals:
            for ts in ts_vals:
                for dmp in damping_vals:
                    specs.append(CandidateSpec(
                        name=f"fast_p{p}_n{n}_ts{ts}_d{dmp}",
                        d1=1.0,
                        par=p,
                        perp=n,
                        time_scale=ts,
                        damping=dmp
                    ))
    for j in [-0.2, 0.2]:
        specs.append(CandidateSpec(f"fast_jerk_{j}", 1.0, 1.0, 0.0, jerk=j))
    return specs

CANDIDATES_SLOW = get_slow_specs()
CANDIDATES_FAST = get_fast_specs()

# Precomputed pseudo-inverse of design matrix for quadratic fit over t = [-4, -3, -2, -1, 0]
P_MATRIX = np.array([
    [ 0.14285715, -0.07142857, -0.14285715, -0.07142857,  0.14285716],
    [ 0.37142855, -0.3857143,  -0.5714286,  -0.18571429,  0.77142864],
    [ 0.08571429, -0.14285716, -0.0857143,   0.25714284,  0.88571435]
], dtype=np.float32)

def extract_smoothed_derivatives(x, window_size=5):
    """
    Fits a 2nd degree polynomial to the last window_size coordinates
    to compute noise-filtered velocity and acceleration vectors at t=0.
    Uses precomputed pseudo-inverse matrix for speed.
    """
    sub_x = x[-window_size:]
    
    coeffs = P_MATRIX @ sub_x
    
    p0_smooth = coeffs[2]
    vel_smooth = coeffs[1]
    acc_smooth = 2.0 * coeffs[0]
    
    # Calculate turn rate using angle between current smoothed velocity and previous smoothed velocity
    # Let's evaluate velocity at t=0 and t=-1
    vel_prev = 2.0 * coeffs[0] * (-1.0) + coeffs[1]
    
    speed = np.linalg.norm(vel_smooth)
    prev_speed = np.linalg.norm(vel_prev)
    
    cross_va = np.cross(vel_smooth, acc_smooth)
    curv = np.linalg.norm(cross_va) / (speed**3 + 1e-6)
    
    cos_theta = np.sum(vel_smooth * vel_prev) / (speed * prev_speed + EPS)
    
    return {
        "smooth_speed": speed,
        "smooth_acc": np.linalg.norm(acc_smooth),
        "smooth_curv": curv,
        "smooth_turn": cos_theta
    }

def make_candidates(x, priors, end_idx=-1, horizon=2):
    """
    Generates candidates by combining:
    1. Speed-adaptive raw physical equations (lowest lag).
    2. Deep learning priors injected directly into the candidate coordinate list.
    """
    p0 = x[end_idx]
    d1 = x[end_idx] - x[end_idx - 1]
    d2 = x[end_idx - 1] - x[end_idx - 2]
    d3 = x[end_idx - 2] - x[end_idx - 3]
    
    acc = d1 - d2
    prev_acc = d2 - d3
    jerk = acc - prev_acc
    
    speed = np.linalg.norm(d1)
    tangent = d1 / (speed + EPS)
    
    acc_par_scalar = np.sum(acc * tangent)
    acc_par = acc_par_scalar * tangent
    acc_perp_vec = acc - acc_par
    
    # Speed threshold: 0.0234 m/s (median speed)
    if speed <= 0.0234:
        cands_list = list(CANDIDATES_SLOW)
    else:
        cands_list = list(CANDIDATES_FAST)
        
    v_scale = horizon
    acc_scale = 0.5 * (horizon**2)
    jerk_scale = (1.0/6.0) * (horizon**3)
    
    dmps = np.array([s.damping for s in cands_list], dtype=np.float32)
    F_v, F_a, F_j = get_damping_factors(dmps)
    
    preds = []
    for i, spec in enumerate(cands_list):
        ts = spec.time_scale
        ts2 = ts ** 2
        ts3 = ts ** 3
        
        fv = F_v[i]
        fa = F_a[i]
        fj = F_j[i]
        
        cand = (p0 
                + spec.d1 * (v_scale * ts * fv) * d1 
                + spec.d2 * (v_scale * ts * fv) * d2 
                + spec.par * (acc_scale * ts2 * fa) * acc_par 
                + spec.perp * (acc_scale * ts2 * fa) * acc_perp_vec
                + spec.jerk * (jerk_scale * ts3 * fj) * jerk)
        preds.append(cand)
        
    preds = np.array(preds, dtype=np.float32)
    
    # 2. Append Prior Candidates (s7_pos, s4_pos)
    s7_pos, s4_pos = priors
    
    # Create Specs for the Priors
    s7_spec = CandidateSpec(name="s7_prior", d1=1.0, par=0.0, perp=0.0)
    s4_spec = CandidateSpec(name="s4_prior", d1=1.0, par=0.0, perp=0.0)
    
    all_cands = np.vstack([preds, s7_pos, s4_pos])
    all_specs = cands_list + [s7_spec, s4_spec]
    
    return all_cands, all_specs
