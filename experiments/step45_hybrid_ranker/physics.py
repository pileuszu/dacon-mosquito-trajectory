import numpy as np
from dataclasses import dataclass

EPS = 1e-8

@dataclass(frozen=True)
class CandidateSpec:
    name: str
    d1: float
    par: float
    perp: float
    binormal: float = 0.0
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
    par_vals = [-0.2, 0.0, 0.2, 0.5]
    perp_vals = [-0.3, -0.1, 0.0, 0.1, 0.3]
    for p in par_vals:
        for n in perp_vals:
            specs.append(CandidateSpec(
                name=f"slow_p{p}_n{n}",
                d1=1.0,
                par=p,
                perp=n,
                binormal=0.0,
                time_scale=1.0,
                damping=0.0
            ))
    return specs

def get_fast_2d_specs():
    specs = []
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
                        binormal=0.0,
                        time_scale=ts,
                        damping=dmp
                    ))
                    
    for j in [-0.2, 0.2]:
        specs.append(CandidateSpec(f"fast_jerk_v_{j}", 1.0, 1.0, 0.0, binormal=0.0, jerk=j))
        
    return specs

def get_fast_3d_specs():
    specs = []
    par_vals = [-0.3, 0.0, 0.6, 1.2, 1.8]
    perp_vals = [-1.2, -0.6, -0.2, 0.0, 0.2, 0.6, 1.2]
    bin_vals = [-0.8, -0.3, 0.0, 0.3, 0.8]
    ts_vals = [0.8, 1.0, 1.2]
    damping_vals = [0.0, 0.3]
    
    for p in par_vals:
        for n in perp_vals:
            for b in bin_vals:
                for ts in ts_vals:
                    for dmp in damping_vals:
                        specs.append(CandidateSpec(
                            name=f"fast_p{p}_n{n}_b{b}_ts{ts}_d{dmp}",
                            d1=1.0,
                            par=p,
                            perp=n,
                            binormal=b,
                            time_scale=ts,
                            damping=dmp
                        ))
                        
    for j in [-0.2, 0.2]:
        specs.append(CandidateSpec(f"fast_jerk_v_{j}", 1.0, 1.0, 0.0, binormal=0.0, jerk=j))
        
    return specs

def get_raw_accel_turning_specs():
    specs = []
    par_vals = [0.0, 0.5, 1.0]
    perp_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    ts_vals = [0.8, 1.0, 1.2]
    damping_vals = [0.0, 0.3]
    for p in par_vals:
        for n in perp_vals:
            for ts in ts_vals:
                for dmp in damping_vals:
                    specs.append(CandidateSpec(
                        name=f"raw_p{p}_n{n}_ts{ts}_d{dmp}",
                        d1=1.0,
                        par=p,
                        perp=n,
                        binormal=0.0,
                        time_scale=ts,
                        damping=dmp
                    ))
    return specs

CANDIDATES_SLOW = get_slow_specs()
CANDIDATES_FAST_2D = get_fast_2d_specs()
CANDIDATES_FAST_3D = get_fast_3d_specs()
CANDIDATES_RAW_ACCEL_TURNING = get_raw_accel_turning_specs()

T5 = np.array([[t**2, t, 1] for t in range(-4, 1)], dtype=np.float32)
PINV_W5_QUAD = np.linalg.pinv(T5)

T5_CUB = np.array([[t**3, t**2, t, 1] for t in range(-4, 1)], dtype=np.float32)
PINV_W5_CUBIC = np.linalg.pinv(T5_CUB)

T3 = np.array([[t**2, t, 1] for t in range(-2, 1)], dtype=np.float32)
PINV_W3_QUAD = np.linalg.pinv(T3)

def extract_rolling_kinematics(x):
    vel = np.diff(x, axis=0)
    speeds = np.linalg.norm(vel, axis=1)
    
    acc = np.diff(vel, axis=0)
    acc_norms = np.linalg.norm(acc, axis=1)
    
    jerk = np.diff(acc, axis=0)
    jerk_norms = np.linalg.norm(jerk, axis=1)
    
    v_norm = speeds[:-1]
    cross_va = np.cross(vel[:-1], acc)
    cross_norm = np.linalg.norm(cross_va, axis=1)
    curvatures = cross_norm / (v_norm**3 + EPS)
    
    curv_rate = curvatures[-1] - curvatures[-2]
    curv_rate_mean_5 = np.mean(np.diff(curvatures)[-5:])
    
    speed_mean_10 = np.mean(speeds[-10:])
    speed_std_10 = np.std(speeds[-10:])
    speed_cv_10 = speed_std_10 / (speed_mean_10 + EPS)
    
    speed_mean_20 = np.mean(speeds[-20:])
    speed_std_20 = np.std(speeds[-20:])
    speed_cv_20 = speed_std_20 / (speed_mean_20 + EPS)
    
    speed_mean_all = np.mean(speeds)
    speed_std_all = np.std(speeds)
    speed_cv_all = speed_std_all / (speed_mean_all + EPS)
    
    acc_mean_20 = np.mean(acc_norms[-20:])
    acc_std_20 = np.std(acc_norms[-20:])
    acc_cv_20 = acc_std_20 / (acc_mean_20 + EPS)
    
    acc_mean_all = np.mean(acc_norms)
    acc_std_all = np.std(acc_norms)
    acc_cv_all = acc_std_all / (acc_mean_all + EPS)
    
    jerk_mean_20 = np.mean(jerk_norms[-20:])
    jerk_std_20 = np.std(jerk_norms[-20:])
    jerk_mean_all = np.mean(jerk_norms)
    jerk_std_all = np.std(jerk_norms)
    
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
        "roll_curv_std_all": curv_std_all,
        
        "ctx_curv_rate": curv_rate,
        "ctx_curv_rate_mean_5": curv_rate_mean_5
    }

def extract_multi_scale_derivatives(x):
    p0 = x[-1]
    
    d1 = x[-1] - x[-2]
    d2 = x[-2] - x[-3]
    d3 = x[-3] - x[-4]
    acc = d1 - d2
    prev_acc = d2 - d3
    jerk = acc - prev_acc
    
    x_w3 = x[-3:]
    coeffs_w3 = PINV_W3_QUAD @ x_w3
    a_w3, b_w3 = coeffs_w3[0], coeffs_w3[1]
    v_w3 = b_w3
    acc_w3 = 2.0 * a_w3
    speed_w3 = np.linalg.norm(v_w3)
    acc_norm_w3 = np.linalg.norm(acc_w3)
    
    cross_w3 = np.cross(v_w3, acc_w3)
    curv_w3 = np.linalg.norm(cross_w3) / (speed_w3**3 + EPS)
    
    v_prev_w3 = v_w3 - acc_w3
    cos_turn_w3 = np.dot(v_w3, v_prev_w3) / (np.linalg.norm(v_w3) * np.linalg.norm(v_prev_w3) + EPS)
    turn_w3 = np.arccos(np.clip(cos_turn_w3, -1.0, 1.0))
    
    x_w5 = x[-5:]
    coeffs_w5 = PINV_W5_QUAD @ x_w5
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
    
    coeffs_cub = PINV_W5_CUBIC @ x_w5
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
    
    CRUISING_CENTROID = np.array([0.0120, 0.0035, 0.0020, 0.0040, 0.0010], dtype=np.float32)
    SACCADIC_CENTROID = np.array([0.0350, 0.0145, 0.0110, 0.0125, 0.0055], dtype=np.float32)
    
    speed_raw = np.linalg.norm(d1)
    tangent_raw = d1 / (speed_raw + EPS)
    feat = np.array([speed_raw, np.linalg.norm(acc), np.linalg.norm(acc - (np.sum(acc * tangent_raw) * tangent_raw)), np.abs(d1[2]), np.abs(acc[2])], dtype=np.float32)
    dist_to_cruising = np.linalg.norm(feat - CRUISING_CENTROID)
    dist_to_saccadic = np.linalg.norm(feat - SACCADIC_CENTROID)
    
    ctx_p_saccade = np.exp(-dist_to_saccadic) / (np.exp(-dist_to_saccadic) + np.exp(-dist_to_cruising) + EPS)
    
    acc_par_scalar = np.sum(acc_w5 * tangent_raw)
    acc_perp_vec = acc_w5 - acc_par_scalar * tangent_raw
    ctx_lat_accel = np.linalg.norm(acc_perp_vec)
    
    jerk_par_scalar = np.sum(jerk * tangent_raw)
    jerk_perp_vec = jerk - jerk_par_scalar * tangent_raw
    ctx_lat_jerk = np.linalg.norm(jerk_perp_vec)
    
    multi_scale_features = {
        "ctx_speed": speed_raw,
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
        "ctx_speed_ratio": speed_raw / (np.linalg.norm(d2) + EPS),
        
        "dist_to_cruising": dist_to_cruising,
        "dist_to_saccadic": dist_to_saccadic,
        "ctx_p_saccade": ctx_p_saccade,
        
        "ctx_lat_accel": ctx_lat_accel,
        "ctx_lat_jerk": ctx_lat_jerk
    }
    
    rolling_features = extract_rolling_kinematics(x)
    
    return {**multi_scale_features, **rolling_features}

def make_candidates(x, priors, end_idx=-1, horizon=2, regime=None):
    from dataclasses import replace
    
    x_sliced = x[:end_idx+1] if end_idx != -1 else x
    p0 = x_sliced[-1]
    d1 = x_sliced[-1] - x_sliced[-2]
    d2 = x_sliced[-2] - x_sliced[-3]
    d3 = x_sliced[-3] - x_sliced[-4]
    
    acc = d1 - d2
    prev_acc = d2 - d3
    jerk = acc - prev_acc
    
    speed = np.linalg.norm(d1)
    tangent = d1 / (speed + EPS)
    
    ctx = extract_multi_scale_derivatives(x_sliced)
    p_sacc = ctx["ctx_p_saccade"]
    ctx_lat_accel = ctx["ctx_lat_accel"]
    ctx_curv = ctx["smooth_curv_w5"]
    
    if regime == "fast_straight_low":
        acc_smooth = acc
        if len(x_sliced) >= 5:
            x_w5 = x_sliced[-5:]
            coeffs_w5 = PINV_W5_QUAD @ x_w5
            acc_smooth = 2.0 * coeffs_w5[0]
            
        cands_base = list(CANDIDATES_FAST_2D)
        S_grid = float(np.clip(1.0 + 0.4 * p_sacc, 1.0, 1.8))
        
    elif regime == "slow_moderate_turning":
        acc_smooth = acc
        if len(x_sliced) >= 5:
            x_w5 = x_sliced[-5:]
            coeffs_w5 = PINV_W5_QUAD @ x_w5
            acc_smooth = 2.0 * coeffs_w5[0]
            
        cands_base = list(CANDIDATES_SLOW)
        S_grid = float(np.clip(1.0 + 0.1 * ctx_curv, 1.0, 1.5))
        
    elif regime == "fast_moderate_turning":
        acc_smooth = acc
        if len(x_sliced) >= 5:
            x_w5 = x_sliced[-5:]
            coeffs_w5 = PINV_W5_QUAD @ x_w5
            acc_smooth = 2.0 * coeffs_w5[0]
            
        cands_base = list(CANDIDATES_FAST_2D)
        S_grid = float(np.clip(1.2 + 0.1 * ctx_curv, 1.2, 2.0))
        
    elif regime == "fast_straight_high":
        acc_smooth = acc
        if len(x_sliced) >= 5:
            x_w5 = x_sliced[-5:]
            coeffs_w5 = PINV_W5_QUAD @ x_w5
            acc_smooth = 2.0 * coeffs_w5[0]
            
        cands_base = list(CANDIDATES_FAST_2D)
        S_grid = float(np.clip(1.0 + 0.6 * p_sacc, 1.0, 3.2))
        
    elif regime == "fast_extreme_turning":
        x_w3 = x_sliced[-3:]
        coeffs_w3 = PINV_W3_QUAD @ x_w3
        acc_smooth = 2.0 * coeffs_w3[0]
        
        speed_w3 = np.linalg.norm(coeffs_w3[1])
        tangent = coeffs_w3[1] / (speed_w3 + EPS)
        
        fallback_specs = [
            CandidateSpec(name="straight_fallback_d0.2", d1=1.0, par=0.0, perp=0.0, binormal=0.0, damping=0.2),
            CandidateSpec(name="straight_fallback_d0.5", d1=1.0, par=0.0, perp=0.0, binormal=0.0, damping=0.5),
            CandidateSpec(name="straight_fallback_d0.8", d1=1.0, par=0.0, perp=0.0, binormal=0.0, damping=0.8),
        ]
        cands_base = list(CANDIDATES_FAST_3D) + fallback_specs
        S_grid = float(np.clip(1.8 + 0.8 * p_sacc, 1.8, 4.2))
        
    elif regime == "slow_extreme_turning":
        x_w3 = x_sliced[-3:]
        coeffs_w3 = PINV_W3_QUAD @ x_w3
        acc_smooth = 2.0 * coeffs_w3[0]
        
        speed_w3 = np.linalg.norm(coeffs_w3[1])
        tangent = coeffs_w3[1] / (speed_w3 + EPS)
        
        cands_base = list(CANDIDATES_FAST_2D)
        S_grid = float(np.clip(1.8 + 0.15 * ctx["smooth_curv_w3"], 1.8, 4.0))
        
    else:
        is_turning = (ctx_curv > 12.0) or (ctx_lat_accel > 0.0020)
        if speed <= 0.0234 and not is_turning:
            cands_base = list(CANDIDATES_SLOW)
            S_grid = 1.0
            acc_smooth = acc
            if len(x_sliced) >= 5:
                x_w5 = x_sliced[-5:]
                coeffs_w5 = PINV_W5_QUAD @ x_w5
                acc_smooth = 2.0 * coeffs_w5[0]
        else:
            cands_base = list(CANDIDATES_FAST_2D)
            S_grid = float(np.clip(1.0 + 0.6 * p_sacc, 1.0, 2.5))
            acc_smooth = acc
            if len(x_sliced) >= 5:
                x_w5 = x_sliced[-5:]
                coeffs_w5 = PINV_W5_QUAD @ x_w5
                acc_smooth = 2.0 * coeffs_w5[0]
                
    acc_par_scalar = np.sum(acc_smooth * tangent)
    acc_par = acc_par_scalar * tangent
    acc_perp_vec = acc_smooth - acc_par
    
    acc_perp_magnitude = np.linalg.norm(acc_perp_vec)
    if acc_perp_magnitude > 1e-6:
        n_hat = acc_perp_vec / acc_perp_magnitude
    else:
        if abs(tangent[0]) > 0.9:
            n_hat = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            n_hat = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        n_hat = n_hat - np.sum(n_hat * tangent) * tangent
        n_hat = n_hat / (np.linalg.norm(n_hat) + EPS)
        
    b_hat = np.cross(tangent, n_hat)
    b_hat = b_hat / (np.linalg.norm(b_hat) + EPS)
    
    acc_binormal_vec = acc_perp_magnitude * b_hat
    
    cands_list = [replace(spec, par=spec.par * S_grid, perp=spec.perp * S_grid, binormal=spec.binormal * S_grid) for spec in cands_base]
        
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
                + spec.binormal * (acc_scale * ts2 * fa) * acc_binormal_vec
                + spec.jerk * (jerk_scale * ts3 * fj) * jerk)
        preds.append(cand)
        
    if regime in ["fast_extreme_turning", "slow_extreme_turning"]:
        acc_par_scalar_raw = np.sum(acc * tangent)
        acc_par_raw = acc_par_scalar_raw * tangent
        acc_perp_vec_raw = acc - acc_par_raw
        
        cands_list_raw = [replace(spec, par=spec.par * S_grid, perp=spec.perp * S_grid, binormal=spec.binormal * S_grid) for spec in CANDIDATES_RAW_ACCEL_TURNING]
        dmps_raw = np.array([s.damping for s in cands_list_raw], dtype=np.float32)
        F_v_raw, F_a_raw, F_j_raw = get_damping_factors(dmps_raw)
        
        preds_raw = []
        for i, spec in enumerate(cands_list_raw):
            ts = spec.time_scale
            ts2 = ts ** 2
            ts3 = ts ** 3
            
            fv = F_v_raw[i]
            fa = F_a_raw[i]
            fj = F_j_raw[i]
            
            cand = (p0 
                    + spec.d1 * (v_scale * ts * fv) * d1 
                    + spec.par * (acc_scale * ts2 * fa) * acc_par_raw 
                    + spec.perp * (acc_scale * ts2 * fa) * acc_perp_vec_raw
                    + spec.jerk * (jerk_scale * ts3 * fj) * jerk)
            preds_raw.append(cand)
            
        preds = preds + preds_raw
        cands_base = cands_base + CANDIDATES_RAW_ACCEL_TURNING
        
    preds = np.array(preds, dtype=np.float32)
    
    s7_pos, s4_pos = priors
    
    s7_spec = CandidateSpec(name="s7_prior", d1=1.0, par=0.0, perp=0.0, binormal=0.0)
    s4_spec = CandidateSpec(name="s4_prior", d1=1.0, par=0.0, perp=0.0, binormal=0.0)
    
    all_cands = np.vstack([preds, s7_pos, s4_pos])
    all_specs = cands_base + [s7_spec, s4_spec]
    
    d_cands = all_cands - p0
    c_speeds = np.linalg.norm(d_cands, axis=-1) / 2.0
    c_speed_ratio = c_speeds / (np.linalg.norm(d1) + EPS)
    
    v0_norm = np.linalg.norm(d1)
    v0_hat = d1 / (v0_norm + EPS)
    d_cands_norm = np.linalg.norm(d_cands, axis=-1)
    d_cands_hat = d_cands / (d_cands_norm[:, None] + EPS)
    
    cos_theta = np.sum(d_cands_hat * v0_hat, axis=-1)
    c_turn_angles = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * (180.0 / np.pi)
    
    hist_turn_deg = float(ctx["smooth_turn_w5"]) * (180.0 / np.pi)
    c_turn_rates = c_turn_angles - hist_turn_deg
    
    c_acc = (all_cands - p0 - 2.0 * d1) / 2.0
    c_accels = np.linalg.norm(c_acc, axis=-1)
    c_acc_par = np.sum(c_acc * v0_hat, axis=-1)[:, None] * v0_hat
    c_acc_perp = c_acc - c_acc_par
    c_lat_accels = np.linalg.norm(c_acc_perp, axis=-1)
    
    c_displacement_normal = np.sum(d_cands * n_hat, axis=-1)
    c_displacement_binormal = np.sum(d_cands * b_hat, axis=-1)
    c_accel_binormal = np.sum(c_acc * b_hat, axis=-1)
    
    c_features = {
        "grid_scale": np.full(len(all_cands), S_grid, dtype=np.float32),
        "cand_speed": c_speeds,
        "cand_speed_ratio": c_speed_ratio,
        "cand_turn_angle": c_turn_angles,
        "cand_turn_rate": c_turn_rates,
        "cand_accel": c_accels,
        "cand_lat_accel": c_lat_accels,
        "cand_disp_normal": c_displacement_normal,
        "cand_disp_binormal": c_displacement_binormal,
        "cand_acc_binormal": c_accel_binormal
    }
    
    return all_cands, all_specs, c_features
