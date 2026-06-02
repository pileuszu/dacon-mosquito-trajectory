import torch
import torch.nn as nn
import numpy as np

# EPS for numerical stability
BaseEPS = 1e-8

def extract_features(X, mean_stats=None, std_stats=None):
    # X shape: (batch_size, 11, 3)
    batch_size = X.shape[0]
    device = X.device
    
    # 1. Compute derivatives
    # Note: dt is 40ms (0.04s) per step
    v = (X[:, 1:] - X[:, :-1]) / 0.04  # (batch_size, 10, 3)
    a = (v[:, 1:] - v[:, :-1]) / 0.04  # (batch_size, 9, 3)
    j = (a[:, 1:] - a[:, :-1]) / 0.04  # (batch_size, 8, 3)
    
    speeds = torch.norm(v, dim=2)  # (batch_size, 10)
    acc_norms = torch.norm(a, dim=2)  # (batch_size, 9)
    jerk_norms = torch.norm(j, dim=2)  # (batch_size, 8)
    
    # 2. Extract features
    f1 = speeds[:, -1:]  # last speed
    f2 = acc_norms[:, -1:]  # last accel norm
    f3 = jerk_norms[:, -1:]  # last jerk norm
    
    f4 = speeds.mean(dim=1, keepdim=True)
    f5 = speeds.std(dim=1, keepdim=True)
    f6 = speeds.max(dim=1, keepdim=True).values
    
    f7 = acc_norms.mean(dim=1, keepdim=True)
    f8 = acc_norms.std(dim=1, keepdim=True)
    f9 = acc_norms.max(dim=1, keepdim=True).values
    
    f10 = jerk_norms.mean(dim=1, keepdim=True)
    f11 = jerk_norms.std(dim=1, keepdim=True)
    f12 = jerk_norms.max(dim=1, keepdim=True).values
    
    pos_std = X.std(dim=1)  # (batch_size, 3)
    f13 = pos_std[:, 0:1]
    f14 = pos_std[:, 1:2]
    f15 = pos_std[:, 2:3]
    
    last_v = v[:, -1]
    f16 = torch.atan2(last_v[:, 1], last_v[:, 0]).unsqueeze(-1)  # yaw angle theta
    f17 = torch.atan2(last_v[:, 2], torch.norm(last_v[:, :2], dim=1) + BaseEPS).unsqueeze(-1)  # pitch angle
    
    last_a = a[:, -1]
    cross_prod = torch.cross(last_v, last_a, dim=1)
    f18 = (torch.norm(cross_prod, dim=1) / (torch.norm(last_v, dim=1) ** 3 + BaseEPS)).unsqueeze(-1)  # curvature
    
    f19 = torch.norm(X[:, -1] - X[:, 0], dim=1, keepdim=True)  # total displacement
    
    f20 = speeds[:, -2:-1]  # speed at t=9
    f21 = speeds[:, -3:-2]  # speed at t=8
    f22 = acc_norms[:, -2:-1]  # accel norm at t=8
    f23 = acc_norms[:, -3:-2]  # accel norm at t=7
    f24 = jerk_norms[:, -2:-1]  # jerk norm at t=7
    
    # 2.1 Z-Dynamics features (from Step 52)
    z_coords = X[:, :, 2]
    v_z = v[:, :, 2]
    a_z = a[:, :, 2]
    
    f25 = z_coords.std(dim=1, keepdim=True)         # Z coordinate variation std
    f26 = v_z.mean(dim=1, keepdim=True)             # Z-axis mean velocity
    f27 = a_z[:, -1:]                               # Z-axis last acceleration
    
    # 2.2 NEW: 3D Frenet Torsion (Torsion tau) feature
    # tau = ((v x a) . j) / (||v x a||^2 + eps)
    last_j = j[:, -1]
    cross_norm_sq = torch.sum(cross_prod ** 2, dim=1, keepdim=True)
    triple_prod = torch.sum(cross_prod * last_j, dim=1, keepdim=True)
    f28 = triple_prod / (cross_norm_sq + BaseEPS)    # Torsion (torsional twist out of plane)
    
    features = torch.cat([
        f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12,
        f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24,
        f25, f26, f27, f28
    ], dim=1)
    
    # 3. Compute scale stats if not provided
    if mean_stats is None or std_stats is None:
        mean_stats = features.mean(dim=0)
        std_stats = features.std(dim=0)
        std_stats[std_stats < 1e-6] = 1.0
        
    features_scaled = (features - mean_stats) / std_stats
    
    # 4. Compute initial R matrix (at t=0ms for history alignment)
    t = last_v / (torch.norm(last_v, dim=1, keepdim=True) + BaseEPS)  # (tangent)
    
    acc_perp = last_a - torch.sum(last_a * t, dim=1, keepdim=True) * t
    n_norm = torch.norm(acc_perp, dim=1, keepdim=True)
    n = acc_perp / (n_norm + BaseEPS)
    
    fallback = torch.zeros_like(n)
    axis = torch.argmin(torch.abs(t), dim=1)
    fallback[torch.arange(batch_size), axis] = 1.0
    fallback = fallback - torch.sum(fallback * t, dim=1, keepdim=True) * t
    fallback = fallback / (torch.norm(fallback, dim=1, keepdim=True) + BaseEPS)
    n = torch.where(n_norm > 1e-6, n, fallback)
    
    b = torch.cross(t, n, dim=1)
    b = b / (torch.norm(b, dim=1, keepdim=True) + BaseEPS)
    R = torch.stack([t, n, b], dim=2)
    
    # 5. Steering indicator classification
    acc_perp_norm = torch.norm(acc_perp, dim=1)
    curv = f18.squeeze(-1)
    is_steering = (curv > 6.0) | (acc_perp_norm > 1.8)
    
    df = X[:, 1:] - X[:, :-1]
    plt_ = X[:, 10, :]
    tht = f16.squeeze(-1)
    spt = f1.squeeze(-1)
    
    return features_scaled, df, plt_, tht, is_steering, last_a, None, R, spt, mean_stats, std_stats
