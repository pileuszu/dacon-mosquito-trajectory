import numpy as np

EPS = 1e-8

def extract_super_features(X):
    """
    Extracts 180+ dimensional biomechanical and geometric features from Cartesian trajectory history.
    X: numpy array of shape [N, seq_len, 3] (e.g., [N, 11, 3] coordinates)
    Returns: numpy array of shape [N, feature_dim]
    """
    N = X.shape[0]
    seq_len = X.shape[1]
    features_list = []
    
    # 1. Coordinate differences & Derivatives
    # dt = 0.04
    v = (X[:, 1:] - X[:, :-1]) / 0.04  # [N, seq_len-1, 3]
    a = (v[:, 1:] - v[:, :-1]) / 0.04  # [N, seq_len-2, 3]
    j = (a[:, 1:] - a[:, :-1]) / 0.04  # [N, seq_len-3, 3]
    s = (j[:, 1:] - j[:, :-1]) / 0.04  # [N, seq_len-4, 3]
    
    # Norms
    speeds = np.linalg.norm(v, axis=2)  
    accs = np.linalg.norm(a, axis=2)    
    jerks = np.linalg.norm(j, axis=2)   
    snaps = np.linalg.norm(s, axis=2)   
    
    # 2. Multi-Scale Sliding Windows (w2, w4, w8)
    # w2 (last 2 frames) - speed & acc
    features_list.extend([
        np.mean(speeds[:, -2:], axis=1), np.std(speeds[:, -2:], axis=1), np.max(speeds[:, -2:], axis=1),
        np.mean(accs[:, -2:], axis=1), np.std(accs[:, -2:], axis=1), np.max(accs[:, -2:], axis=1)
    ])
    
    # w4 (last 4 frames)
    features_list.extend([
        np.mean(speeds[:, -4:], axis=1), np.std(speeds[:, -4:], axis=1), np.max(speeds[:, -4:], axis=1), np.min(speeds[:, -4:], axis=1),
        np.mean(accs[:, -4:], axis=1), np.std(accs[:, -4:], axis=1), np.max(accs[:, -4:], axis=1), np.min(accs[:, -4:], axis=1),
        np.mean(jerks[:, -4:], axis=1), np.std(jerks[:, -4:], axis=1), np.max(jerks[:, -4:], axis=1), np.min(jerks[:, -4:], axis=1)
    ])
    
    # w8 (all available steps)
    features_list.extend([
        np.mean(speeds, axis=1), np.std(speeds, axis=1), np.max(speeds, axis=1), np.min(speeds, axis=1),
        np.mean(accs, axis=1), np.std(accs, axis=1), np.max(accs, axis=1), np.min(accs, axis=1),
        np.mean(jerks, axis=1), np.std(jerks, axis=1), np.max(jerks, axis=1), np.min(jerks, axis=1),
        np.mean(snaps, axis=1), np.std(snaps, axis=1), np.max(snaps, axis=1), np.min(snaps, axis=1)
    ])
    
    # Raw sequential speeds, accs, jerks, snaps (dynamic dimensions)
    for t_idx in range(speeds.shape[1]):
        features_list.append(speeds[:, t_idx])
    for t_idx in range(accs.shape[1]):
        features_list.append(accs[:, t_idx])
    for t_idx in range(jerks.shape[1]):
        features_list.append(jerks[:, t_idx])
    for t_idx in range(snaps.shape[1]):
        features_list.append(snaps[:, t_idx])
        
    # 3. High-Order Derivatives on Frenet Frame (last frame)
    v_last = v[:, -1]
    a_last = a[:, -1]
    j_last = j[:, -1]
    s_last = s[:, -1]
    
    speed_last = speeds[:, -1]
    
    # Tangent vector T
    T = v_last / (speed_last[:, None] + EPS)
    
    # Normal vector N
    a_par_scalar = np.sum(a_last * T, axis=1)
    a_perp = a_last - a_par_scalar[:, None] * T
    a_perp_norm = np.linalg.norm(a_perp, axis=1)
    
    N = a_perp / (a_perp_norm[:, None] + EPS)
    fallback = np.zeros_like(N)
    axis = np.argmin(np.abs(T), axis=1)
    fallback[np.arange(N.shape[0]), axis] = 1.0
    fallback = fallback - np.sum(fallback * T, axis=1)[:, None] * T
    fallback = fallback / (np.linalg.norm(fallback, axis=1)[:, None] + EPS)
    N = np.where(a_perp_norm[:, None] > 1e-6, N, fallback)
    
    # Binormal vector B
    B = np.cross(T, N, axis=1)
    B = B / (np.linalg.norm(B, axis=1)[:, None] + EPS)
    
    # Project derivatives
    a_T = a_par_scalar
    a_N = np.sum(a_last * N, axis=1)
    a_B = np.sum(a_last * B, axis=1)
    
    j_T = np.sum(j_last * T, axis=1)
    j_N = np.sum(j_last * N, axis=1)
    j_B = np.sum(j_last * B, axis=1)
    
    s_T = np.sum(s_last * T, axis=1)
    s_N = np.sum(s_last * N, axis=1)
    s_B = np.sum(s_last * B, axis=1)
    
    features_list.extend([
        a_T, a_N, a_B, a_perp_norm,
        j_T, j_N, j_B, np.linalg.norm(j_last, axis=1),
        s_T, s_N, s_B, np.linalg.norm(s_last, axis=1)
    ])
    
    # 4. Aerodynamics & Inertial Biomechanics
    f_rayleigh = speed_last[:, None]**2 * v_last
    f_rayleigh_norm = np.linalg.norm(f_rayleigh, axis=1)
    f_linear = speed_last[:, None] * v_last
    f_linear_norm = np.linalg.norm(f_linear, axis=1)
    
    ke_last = speed_last**2
    ke_prev = speeds[:, -2]**2
    d_ke = ke_last - ke_prev
    
    cross_prod = np.cross(v_last, a_last, axis=1)
    cross_norm = np.linalg.norm(cross_prod, axis=1)
    curv_last = cross_norm / (speed_last**3 + EPS)
    
    acc_centripetal = curv_last * (speed_last**2)
    
    features_list.extend([
        f_rayleigh[:, 0], f_rayleigh[:, 1], f_rayleigh[:, 2], f_rayleigh_norm,
        f_linear[:, 0], f_linear[:, 1], f_linear[:, 2], f_linear_norm,
        ke_last, d_ke, curv_last, acc_centripetal
    ])
    
    # 5. Curvature & Torsion Profile
    curv_seq = []
    torsion_seq = []
    
    # Sequential curvature & torsion (excluding the last one which is processed separately)
    for t_idx in range(2, v.shape[1] - 1):
        vt = v[:, t_idx]
        at = a[:, t_idx-1]
        cp = np.cross(vt, at, axis=1)
        cp_norm = np.linalg.norm(cp, axis=1)
        curv_t = cp_norm / (np.linalg.norm(vt, axis=1)**3 + EPS)
        curv_seq.append(curv_t)
        features_list.append(curv_t)
        
    for t_idx in range(3, v.shape[1] - 1):
        vt = v[:, t_idx]
        at = a[:, t_idx-1]
        jt = j[:, t_idx-2]
        cp = np.cross(vt, at, axis=1)
        cp_norm_sq = np.sum(cp**2, axis=1)
        torsion_t = np.sum(cp * jt, axis=1) / (cp_norm_sq + EPS)
        torsion_seq.append(torsion_t)
        features_list.append(torsion_t)
        
    curv_seq = np.stack(curv_seq, axis=1)  
    torsion_seq = np.stack(torsion_seq, axis=1)  
    
    d_curv = curv_last - curv_seq[:, -1]
    features_list.extend([
        np.mean(curv_seq, axis=1), np.std(curv_seq, axis=1), np.max(curv_seq, axis=1), d_curv,
        np.mean(torsion_seq, axis=1), np.std(torsion_seq, axis=1), np.max(torsion_seq, axis=1)
    ])
    
    # 6. Center of Mass Relative Coordinates
    weights = np.arange(1, seq_len + 1, dtype=float)
    weights /= np.sum(weights)
    p_com = np.sum(X * weights[None, :, None], axis=1)  # [N, 3]
    
    p_last = X[:, -1]
    r_com = p_last - p_com  # [N, 3]
    r_com_norm = np.linalg.norm(r_com, axis=1)
    
    theta_com = np.arctan2(r_com[:, 1], r_com[:, 0])
    phi_com = np.arccos(np.clip(r_com[:, 2] / (r_com_norm + EPS), -1.0, 1.0))
    
    v_com = v_last - np.mean(v, axis=1)
    a_com = a_last - np.mean(a, axis=1)
    
    features_list.extend([
        r_com[:, 0], r_com[:, 1], r_com[:, 2], r_com_norm,
        theta_com, phi_com,
        v_com[:, 0], v_com[:, 1], v_com[:, 2], np.linalg.norm(v_com, axis=1),
        a_com[:, 0], a_com[:, 1], a_com[:, 2], np.linalg.norm(a_com, axis=1)
    ])
    
    # 7. Volatility, Saccade & Turning Indicators
    volatility = np.std(speeds, axis=1) / (np.mean(speeds, axis=1) + EPS)
    is_steering_soft = curv_last * a_perp_norm
    
    features_list.extend([
        volatility, is_steering_soft
    ])
    
    features_mat = np.column_stack(features_list)
    return features_mat
