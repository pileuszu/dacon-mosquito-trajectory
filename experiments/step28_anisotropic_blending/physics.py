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
            F_v = (1.0 - np.exp( -lam)) / lam
            F_a = 2.0 * (lam - 1.0 + np.exp( -lam)) / (lam**2)
            F_j = 6.0 * (0.5*(lam**2) - lam + 1.0 - np.exp( -lam)) / (lam**3)
            
    return F_v, F_a, F_j

def get_slow_specs():
    specs = []
    # 4 * 5 = 20 slow candidates
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
    # 8 * 11 * 5 * 3 = 1320 candidates (damping expanded to 0.0, 0.2, 0.5)
    par_vals = [-0.5, 0.0, 0.4, 0.7, 1.0, 1.3, 1.6, 2.0]
    perp_vals = [-1.5, -1.0, -0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6, 1.0, 1.5]
    ts_vals = [0.7, 0.9, 1.0, 1.1, 1.3]
    damping_vals = [0.0, 0.2, 0.5]
    
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
        specs.append(CandidateSpec(f"fast_jerk_v_{j}", 1.0, 1.0, 0.0, jerk=j))
        
    return specs

CANDIDATES_SLOW = get_slow_specs()
CANDIDATES_FAST = get_fast_specs()

# Precompute W5 Quadratic Pseudo-Inverse
T5 = np.array([[t**2, t, 1] for t in range(-4, 1)], dtype=np.float32)
PINV_W5_QUAD = np.linalg.pinv(T5) # Shape (3, 5)

# Precompute W5 Cubic Pseudo-Inverse
T5_CUB = np.array([[t**3, t**2, t, 1] for t in range(-4, 1)], dtype=np.float32)
PINV_W5_CUBIC = np.linalg.pinv(T5_CUB) # Shape (4, 5)

def extract_rolling_kinematics(x):
    """
    Compute rolling statistical features of velocities, accelerations, and curvatures
    over the 400ms historical trajectory.
    """
    vel = np.diff(x, axis=0) # shape (L-1, 3)
    speeds = np.linalg.norm(vel, axis=1) # shape (L-1,)
    
    acc = np.diff(vel, axis=0) # shape (L-2, 3)
    acc_norms = np.linalg.norm(acc, axis=1) # shape (L-2,)
    
    jerk = np.diff(acc, axis=0) # shape (L-3, 3)
    jerk_norms = np.linalg.norm(jerk, axis=1) # shape (L-3,)
    
    # Curvature at each step
    v_norm = speeds[:-1] # shape (L-2,)
    cross_va = np.cross(vel[:-1], acc) # shape (L-2, 3)
    cross_norm = np.linalg.norm(cross_va, axis=1)
    curvatures = cross_norm / (v_norm**3 + EPS)
    
    # 1. Speeds over different window horizons
    speed_mean_10 = np.mean(speeds[-10:])
    speed_std_10 = np.std(speeds[-10:])
    speed_cv_10 = speed_std_10 / (speed_mean_10 + EPS)
    
    speed_mean_20 = np.mean(speeds[-20:])
    speed_std_20 = np.std(speeds[-20:])
    speed_cv_20 = speed_std_20 / (speed_mean_20 + EPS)
    
    speed_mean_all = np.mean(speeds)
    speed_std_all = np.std(speeds)
    speed_cv_all = speed_std_all / (speed_mean_all + EPS)
    
    # 2. Acceleration statistics
    acc_mean_20 = np.mean(acc_norms[-20:])
    acc_std_20 = np.std(acc_norms[-20:])
    acc_cv_20 = acc_std_20 / (acc_mean_20 + EPS)
    
    acc_mean_all = np.mean(acc_norms)
    acc_std_all = np.std(acc_norms)
    acc_cv_all = acc_std_all / (acc_mean_all + EPS)
    
    # 3. Jerk statistics
    jerk_mean_20 = np.mean(jerk_norms[-20:])
    jerk_std_20 = np.std(jerk_norms[-20:])
    jerk_mean_all = np.mean(jerk_norms)
    jerk_std_all = np.std(jerk_norms)
    
    # 4. Curvature statistics
    curv_mean_20 = np.mean(curvatures[-20:])
    curv_std_20 = np.std(curvatures[-20:])
    curv_mean_all = np.mean(curvatures)
    curv_std_all = np.std(curvatures)
    
    return {
        "roll_speed_mean_10": speed_mean_10,
        "roll_speed_std_10": speed_std_10,
        "roll_speed_cv_10": speed_cv_10,
        
        "roll_speed_mean_20": speed_mean_20,
        "roll_speed_std_20": speed_std_20,
        "roll_speed_cv_20": speed_cv_20,
        
        "roll_speed_mean_all": speed_mean_all,
        "roll_speed_std_all": speed_std_all,
        "roll_speed_cv_all": speed_cv_all,
        
        "roll_acc_mean_20": acc_mean_20,
        "roll_acc_std_20": acc_std_20,
        "roll_acc_cv_20": acc_cv_20,
        
        "roll_acc_mean_all": acc_mean_all,
        "roll_acc_std_all": acc_std_all,
        "roll_acc_cv_all": acc_cv_all,
        
        "roll_jerk_mean_20": jerk_mean_20,
        "roll_jerk_std_20": jerk_std_20,
        "roll_jerk_mean_all": jerk_mean_all,
        "roll_jerk_std_all": jerk_std_all,
        
        "roll_curv_mean_20": curv_mean_20,
        "roll_curv_std_20": curv_std_20,
        "roll_curv_mean_all": curv_mean_all,
        "roll_curv_std_all": curv_std_all
    }

def extract_multi_scale_derivatives(x):
    """
    x: shape (L, 3) - 3D trajectory context
    Extract W3-Quadratic, W5-Quadratic, and W5-Cubic smoothed derivatives.
    """
    p0 = x[-1]
    
    # 1. Raw Derivatives
    d1 = x[-1] - x[-2]
    d2 = x[-2] - x[-3]
    d3 = x[-3] - x[-4]
    acc = d1 - d2
    prev_acc = d2 - d3
    jerk = acc - prev_acc
    
    # 2. W3 Quadratic Fit (Last 3 points)
    x_w3 = x[-3:]
    a_w3 = 0.5 * x_w3[0] - x_w3[1] + 0.5 * x_w3[2]
    b_w3 = 0.5 * x_w3[0] - 2.0 * x_w3[1] + 1.5 * x_w3[2]
    
    v_w3 = b_w3
    acc_w3 = 2.0 * a_w3
    speed_w3 = np.linalg.norm(v_w3)
    acc_norm_w3 = np.linalg.norm(acc_w3)
    
    cross_w3 = np.cross(v_w3, acc_w3)
    curv_w3 = np.linalg.norm(cross_w3) / (speed_w3**3 + EPS)
    
    v_prev_w3 = v_w3 - acc_w3
    cos_turn_w3 = np.dot(v_w3, v_prev_w3) / (np.linalg.norm(v_w3) * np.linalg.norm(v_prev_w3) + EPS)
    turn_w3 = np.arccos(np.clip(cos_turn_w3, -1.0, 1.0))
    
    # 3. W5 Quadratic Fit
    x_w5 = x[-5:]
    coeffs_w5 = PINV_W5_QUAD @ x_w5 # shape (3, 3) -> a_w5, b_w5, c_w5
    a_w5, b_w5 = coeffs_w5[0], coeffs_w5[1]
    
    v_w5 = b_w5
    acc_w5 = 2.0 * a_w5
    speed_w5 = np.linalg.norm(v_w5)
    acc_norm_w5 = np.linalg.norm(acc_w5)
    
    cross_w5 = np.cross(v_w5, acc_w5)
    curv_w5 = np.linalg.norm(cross_w5) / (speed_w5**3 + EPS)
    
    v_prev_w5 = v_w5 - acc_w5
    cos_turn_w5 = np.dot(v_w5, v_prev_w5) / (np.linalg.norm(v_w5) * np.linalg.norm(v_prev_w5) + EPS)
    turn_w5 = np.arccos(np.clip(cos_turn_w5, -1.0, 1.0))
    
    # 4. W5 Cubic Fit
    coeffs_cub = PINV_W5_CUBIC @ x_w5 # shape (4, 3) -> a_cub, b_cub, c_cub, d_cub
    a_cub, b_cub, c_cub = coeffs_cub[0], coeffs_cub[1], coeffs_cub[2]
    
    v_cub = c_cub
    acc_cub = 2.0 * b_cub
    jerk_cub = 6.0 * a_cub
    
    speed_cub = np.linalg.norm(v_cub)
    acc_norm_cub = np.linalg.norm(acc_cub)
    jerk_norm_cub = np.linalg.norm(jerk_cub)
    
    cross_cub = np.cross(v_cub, acc_cub)
    curv_cub = np.linalg.norm(cross_cub) / (speed_cub**3 + EPS)
    
    v_prev_cub = v_cub - acc_cub
    cos_turn_cub = np.dot(v_cub, v_prev_cub) / (np.linalg.norm(v_cub) * np.linalg.norm(v_prev_cub) + EPS)
    turn_cub = np.arccos(np.clip(cos_turn_cub, -1.0, 1.0))
    
    # Behavior gating Centroids
    CRUISING_CENTROID = np.array([0.0120, 0.0035, 0.0020, 0.0040, 0.0010], dtype=np.float32)
    SACCADIC_CENTROID = np.array([0.0350, 0.0145, 0.0110, 0.0125, 0.0055], dtype=np.float32)
    
    # Speed and accelerations to compare with centroids
    feat = np.array([np.linalg.norm(d1), np.linalg.norm(acc), np.linalg.norm(acc - (np.sum(acc * d1 / (np.linalg.norm(d1)+EPS)) * d1 / (np.linalg.norm(d1)+EPS))), np.abs(d1[2]), np.abs(acc[2])], dtype=np.float32)
    dist_to_cruising = np.linalg.norm(feat - CRUISING_CENTROID)
    dist_to_saccadic = np.linalg.norm(feat - SACCADIC_CENTROID)
    
    ctx_p_saccade = np.exp(-dist_to_saccadic) / (np.exp(-dist_to_saccadic) + np.exp(-dist_to_cruising) + EPS)
    
    multi_scale_features = {
        "ctx_speed": np.linalg.norm(d1),
        "ctx_acc": np.linalg.norm(acc),
        "ctx_jerk": np.linalg.norm(jerk),
        
        "smooth_speed_w3": speed_w3,
        "smooth_acc_w3": acc_norm_w3,
        "smooth_curv_w3": curv_w3,
        "smooth_turn_w3": turn_w3,
        
        "smooth_speed_w5": speed_w5,
        "smooth_acc_w5": acc_norm_w5,
        "smooth_curv_w5": curv_w5,
        "smooth_turn_w5": turn_w5,
        
        "smooth_speed_cub": speed_cub,
        "smooth_acc_cub": acc_norm_cub,
        "smooth_curv_cub": curv_cub,
        "smooth_turn_cub": turn_cub,
        "smooth_jerk_cub": jerk_norm_cub,
        
        "ctx_z_vel": d1[2],
        "ctx_z_acc": acc[2],
        "ctx_speed_ratio": np.linalg.norm(d1) / (np.linalg.norm(d2) + EPS),
        
        "dist_to_cruising": dist_to_cruising,
        "dist_to_saccadic": dist_to_saccadic,
        "ctx_p_saccade": ctx_p_saccade
    }
    
    # Extract rolling stats
    rolling_features = extract_rolling_kinematics(x)
    
    return {**multi_scale_features, **rolling_features}

def make_candidates(x, priors, end_idx=-1, horizon=2):
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
    
    s7_pos, s4_pos = priors
    
    s7_spec = CandidateSpec(name="s7_prior", d1=1.0, par=0.0, perp=0.0)
    s4_spec = CandidateSpec(name="s4_prior", d1=1.0, par=0.0, perp=0.0)
    
    all_cands = np.vstack([preds, s7_pos, s4_pos])
    all_specs = cands_list + [s7_spec, s4_spec]
    
    return all_cands, all_specs
