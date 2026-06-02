import numpy as np

EPS = 1e-8

def make_candidates_physical(x: np.ndarray, end_idx: int = 10) -> np.ndarray:
    """
    Generates 27 physical candidates using predefined biomechanical specification rules.
    x: numpy array of shape [N, seq_len, 3] (e.g. seq_len=11)
    """
    p0 = x[:, end_idx]
    d1 = x[:, end_idx] - x[:, end_idx - 1]
    d2 = x[:, end_idx - 1] - x[:, end_idx - 2]
    acc = d1 - d2
    
    prev_acc = d2 - (x[:, end_idx - 2] - x[:, end_idx - 3])
    jerk = acc - prev_acc
    
    speeds = np.linalg.norm(d1, axis=1, keepdims=True)
    tangent = d1 / (speeds + EPS)
    acc_par = np.sum(acc * tangent, axis=1, keepdims=True) * tangent
    acc_perp = acc - acc_par
    
    horizon = 2
    v_scale = horizon / 2.0
    acc_scale = (horizon / 2.0) ** 2
    
    # Predefined physical parameter specs: (d1, par, perp, d2, jerk, time_scale)
    specs = [
        (2.00, 0.00, 0.00, 0.0, 0.0, 1.0),
        (2.00, 0.40, 0.40, 0.0, 0.0, 1.0),
        (2.00, 0.50, 0.50, 0.0, 0.0, 1.0),
        (1.98, 0.56, 0.56, 0.0, 0.0, 1.0),
        (2.00, 0.60, 0.60, 0.0, 0.0, 1.0),
        (1.98, 0.96, -0.08, 0.0, 0.0, 1.0),
        (1.98, 0.90, 0.00, 0.0, 0.0, 1.0),
        (1.98, 1.00, 0.00, 0.0, 0.0, 1.0),
        (2.00, 1.00, -0.10, 0.0, 0.0, 1.0),
        (1.96, 0.90, 0.20, 0.0, 0.0, 1.0),
        (2.02, 0.80, 0.20, 0.0, 0.0, 1.0),
        (1.94, 1.10, -0.20, 0.0, 0.0, 1.0),
        (2.06, 1.00, -0.08, 0.0, 0.0, 1.0),
        (1.90, 1.00, -0.08, 0.0, 0.0, 1.0),
        (1.98, 0.80, -0.05, 0.0, 0.08, 1.0),
        (1.98, 0.80, -0.05, 0.0, -0.08, 1.0),
        (1.98, 0.70, -0.20, 0.0, 0.0, 1.0),
        (1.98, 1.20, -0.20, 0.0, 0.0, 1.0),
        (1.98, 1.20, 0.20, 0.0, 0.0, 1.0),
        (2.08, 1.20, -0.20, 0.0, 0.0, 1.0),
        (1.86, 0.70, 0.20, 0.0, 0.0, 1.0),
        # Time-scaled physical variants
        (1.98, 0.96, -0.08, 0.0, 0.0, 0.85),
        (1.98, 0.96, -0.08, 0.0, 0.0, 0.92),
        (1.98, 0.96, -0.08, 0.0, 0.0, 1.08),
        (1.98, 0.96, -0.08, 0.0, 0.0, 1.15),
        (1.98, 1.10, -0.20, 0.0, 0.0, 1.10),
        (1.96, 0.90, 0.20, 0.0, 0.0, 0.90),
    ]
    
    preds = []
    for spec in specs:
        spec_v_scale = v_scale * spec[5]
        spec_acc_scale = acc_scale * (spec[5] ** 2)
        preds.append(
            p0
            + spec[0] * spec_v_scale * d1
            + spec[3] * spec_v_scale * d2
            + spec[1] * spec_acc_scale * acc_par
            + spec[2] * spec_acc_scale * acc_perp
            + spec[4] * spec_acc_scale * jerk
        )
    return np.stack(preds, axis=1).astype(np.float32)

def generate_hybrid_36_grid(x: np.ndarray, ode_preds: np.ndarray) -> np.ndarray:
    """
    Combines 27 physical candidates with 1 ODE prediction and 8 micro-offsets around it.
    """
    N = len(x)
    phys_cand = make_candidates_physical(x, 10)  # [N, 27, 3]
    
    # Compute Frenet frame at the end of the history
    d1 = x[:, -1] - x[:, -2]
    speeds = np.linalg.norm(d1, axis=1, keepdims=True)
    t = d1 / (speeds + EPS)
    
    n = np.zeros_like(t)
    axis = np.argmin(np.abs(t), axis=1)
    n[np.arange(N), axis] = 1.0
    n = n - np.sum(n * t, axis=1, keepdims=True) * t
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + EPS)
    b = np.cross(t, n, axis=1)
    
    # 8 local micro-offsets
    offsets = [
        (0.005, 0.005), (0.005, -0.005), (-0.005, 0.005), (-0.005, -0.005),
        (0.005, 0.0), (-0.005, 0.0), (0.0, 0.005), (0.0, -0.005)
    ]
    
    ode_offsets = []
    for dn, db in offsets:
        off_pt = ode_preds + dn * n + db * b
        ode_offsets.append(off_pt)
        
    ode_offsets_arr = np.stack(ode_offsets, axis=1)  # [N, 8, 3]
    
    # Combine: 27 Physical + 1 Raw ODE + 8 ODE Offsets = 36 total candidates
    hybrid_grid = np.concatenate([phys_cand, ode_preds[:, None, :], ode_offsets_arr], axis=1)
    return hybrid_grid

def get_dynamic_delta(speed, curvature, acc_perp_norm):
    """
    Get dynamic scaling factor delta for grid expansion.
    """
    is_steering = (curvature > 6.0) | (acc_perp_norm > 1.8)
    if is_steering:
        delta = 0.012
    elif speed <= 0.50:
        delta = 0.005 * (speed / 0.36)
    else:
        delta = 0.008 * (speed / 0.80)
    return np.clip(delta, 0.001, 0.025)

def get_frenet_unit_vectors(x_data):
    """
    Extracts Frenet frame unit vectors (Tangent T, Normal N, Binormal B) at the last frame.
    """
    v = (x_data[:, 1:] - x_data[:, :-1]) / 0.04
    a = (v[:, 1:] - v[:, :-1]) / 0.04
    v_last = v[:, -1]
    a_last = a[:, -1]
    
    speeds = np.linalg.norm(v_last, axis=1)
    T = v_last / (speeds[:, None] + EPS)
    a_par = np.sum(a_last * T, axis=1)
    a_perp = a_last - a_par[:, None] * T
    a_perp_norm = np.linalg.norm(a_perp, axis=1)
    N = a_perp / (a_perp_norm[:, None] + EPS)
    
    fallback = np.zeros_like(N)
    axis = np.argmin(np.abs(T), axis=1)
    fallback[np.arange(N.shape[0]), axis] = 1.0
    fallback = fallback - np.sum(fallback * T, axis=1)[:, None] * T
    fallback = fallback / (np.linalg.norm(fallback, axis=1)[:, None] + EPS)
    N = np.where(a_perp_norm[:, None] > 1e-6, N, fallback)
    
    B = np.cross(T, N, axis=1)
    B = B / (np.linalg.norm(B, axis=1)[:, None] + EPS)
    
    return T, N, B, speeds, a_perp_norm

def generate_hybrid_43_grid(x: np.ndarray, cfm_preds: np.ndarray, base_36_candidates: np.ndarray) -> np.ndarray:
    """
    Generates 43 hybrid candidates by augmenting the 36-candidate grid with the CFM predictions
    and Frenet-aligned offsets.
    """
    N = len(x)
    T_vec, N_vec, B_vec, speeds, a_perp_norm = get_frenet_unit_vectors(x)
    
    # Calculate curvature
    v = (x[:, 1:] - x[:, :-1]) / 0.04
    a = (v[:, 1:] - v[:, :-1]) / 0.04
    v_last = v[:, -1]
    a_last = a[:, -1]
    cross_prod = np.cross(v_last, a_last, axis=1)
    cross_norm = np.linalg.norm(cross_prod, axis=1)
    curv = cross_norm / (speeds ** 3 + EPS)
    
    candidates_43 = np.zeros((N, 43, 3), dtype=np.float32)
    
    for idx in range(N):
        p_last = x[idx, -1]
        T = T_vec[idx]
        N_dir = N_vec[idx]
        B = B_vec[idx]
        
        speed = speeds[idx]
        c_val = curv[idx]
        ap = a_perp_norm[idx]
        
        delta = get_dynamic_delta(speed, c_val, ap)
        cfm_pt = cfm_preds[idx]
        cands_36 = base_36_candidates[idx]
        
        # Build 7 Extra CFM-guided candidates dynamically
        extra_cands = np.array([
            cfm_pt,
            cfm_pt + delta * T,
            cfm_pt - delta * T,
            cfm_pt + delta * N_dir,
            cfm_pt - delta * N_dir,
            cfm_pt + delta * B,
            cfm_pt - delta * B
        ])
        
        candidates_43[idx] = np.vstack([cands_36, extra_cands])
        
    return candidates_43
