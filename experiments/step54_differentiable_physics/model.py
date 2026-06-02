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
    
    # 2.1 Z-Dynamics features
    z_coords = X[:, :, 2]
    v_z = v[:, :, 2]
    a_z = a[:, :, 2]
    
    f25 = z_coords.std(dim=1, keepdim=True)         # Z coordinate variation std
    f26 = v_z.mean(dim=1, keepdim=True)             # Z-axis mean velocity
    f27 = a_z[:, -1:]                               # Z-axis last acceleration
    
    # 2.2 3D Frenet Torsion (Torsion tau) feature
    last_j = j[:, -1]
    cross_norm_sq = torch.sum(cross_prod ** 2, dim=1, keepdim=True)
    triple_prod = torch.sum(cross_prod * last_j, dim=1, keepdim=True)
    f28 = triple_prod / (cross_norm_sq + BaseEPS)    # Torsion (torsional twist out of plane)
    
    # 2.3 NEW: Frenet Jerk Projections (f29, f30, f31)
    t = last_v / (torch.norm(last_v, dim=1, keepdim=True) + BaseEPS)  # Tangent
    acc_perp = last_a - torch.sum(last_a * t, dim=1, keepdim=True) * t
    n_norm = torch.norm(acc_perp, dim=1, keepdim=True)
    n = acc_perp / (n_norm + BaseEPS)
    
    # Fallback normal if acceleration is parallel to velocity
    fallback = torch.zeros_like(n)
    axis = torch.argmin(torch.abs(t), dim=1)
    fallback[torch.arange(batch_size), axis] = 1.0
    fallback = fallback - torch.sum(fallback * t, dim=1, keepdim=True) * t
    fallback = fallback / (torch.norm(fallback, dim=1, keepdim=True) + BaseEPS)
    n = torch.where(n_norm > 1e-6, n, fallback)
    
    b = torch.cross(t, n, dim=1)
    b = b / (torch.norm(b, dim=1, keepdim=True) + BaseEPS)
    
    f29 = torch.sum(last_j * t, dim=1, keepdim=True) # J_T
    f30 = torch.sum(last_j * n, dim=1, keepdim=True) # J_N
    f31 = torch.sum(last_j * b, dim=1, keepdim=True) # J_B
    
    # 2.4 NEW: Torsion Volatility & Curvature Volatility (f32, f33)
    # Reconstruct whole sequence of torsion and curvature
    cross_prod_seq = torch.cross(v[:, 1:-1], a[:, 1:], dim=2) # (batch_size, 8, 3)
    cross_norm_sq_seq = torch.sum(cross_prod_seq ** 2, dim=2) # (batch_size, 8)
    triple_prod_seq = torch.sum(cross_prod_seq * j, dim=2) # (batch_size, 8)
    torsion_seq = triple_prod_seq / (cross_norm_sq_seq + BaseEPS) # (batch_size, 8)
    f32 = torsion_seq.std(dim=1, keepdim=True) # Torsion volatility
    
    curv_seq = torch.norm(cross_prod_seq, dim=2) / (torch.norm(v[:, 1:-1], dim=2) ** 3 + BaseEPS) # (batch_size, 8)
    f33 = curv_seq.std(dim=1, keepdim=True) # Curvature volatility
    
    # 2.5 NEW: Angular Kinematics (f34, f35)
    # omega = ||v x a|| / ||v||^2
    omega_last = torch.norm(cross_prod, dim=1, keepdim=True) / (torch.norm(last_v, dim=1, keepdim=True) ** 2 + BaseEPS)
    f34 = omega_last
    
    prev_cross = torch.cross(v[:, -2], a[:, -2], dim=1)
    omega_prev = torch.norm(prev_cross, dim=1, keepdim=True) / (torch.norm(v[:, -2], dim=1, keepdim=True) ** 2 + BaseEPS)
    f35 = (omega_last - omega_prev) / 0.04 # Angular acceleration approximation
    
    # 2.6 NEW: Volatility & Directional features (f36, f37, f38)
    f36 = speeds[:, -1:] / (speeds[:, -2:-1] + BaseEPS) # Speed Decay Ratio
    
    cos_theta_a = torch.sum(a[:, -1] * a[:, -2], dim=1, keepdim=True) / (acc_norms[:, -1:] * acc_norms[:, -2:-1] + BaseEPS)
    cos_theta_a = torch.clamp(cos_theta_a, -1.0, 1.0)
    f37 = torch.acos(cos_theta_a) # Angle of acceleration shift
    
    f38 = jerk_norms.std(dim=1, keepdim=True) # Jerk volatility norm
    
    # Concatenate all 38 features
    features = torch.cat([
        f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12,
        f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24,
        f25, f26, f27, f28, f29, f30, f31, f32, f33, f34, f35, f36, f37, f38
    ], dim=1)
    
    # 3. Compute scale stats if not provided
    if mean_stats is None or std_stats is None:
        mean_stats = features.mean(dim=0)
        std_stats = features.std(dim=0)
        std_stats = torch.where(std_stats < 1e-6, torch.ones_like(std_stats), std_stats)
        
    features_scaled = (features - mean_stats) / std_stats

    
    # 4. Alignment R matrix
    R = torch.stack([t, n, b], dim=2)
    
    # 5. Steering indicator
    acc_perp_norm = torch.norm(acc_perp, dim=1)
    curv = f18.squeeze(-1)
    is_steering = (curv > 6.0) | (acc_perp_norm > 1.8)
    
    df = X[:, 1:] - X[:, :-1]
    plt_ = X[:, 10, :]
    tht = f16.squeeze(-1)
    spt = f1.squeeze(-1)
    
    return features_scaled, df, plt_, tht, is_steering, last_a, None, R, spt, mean_stats, std_stats


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )
        self.ln = nn.LayerNorm(dim)
        
    def forward(self, x):
        return self.ln(x + self.net(x))


class SpatioTemporalAttention(nn.Module):
    def __init__(self, feature_dim=38, latent_dim=64, num_candidates=36):
        super().__init__()
        self.num_candidates = num_candidates
        self.latent_dim = latent_dim
        
        # Temporal Encoder: Processes the 10-step history velocities/positions
        self.temporal_gru = nn.GRU(input_size=3, hidden_size=latent_dim, num_layers=1, batch_first=True)
        
        # Projection layer for scaled context features (38D)
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        
        # Spatial Encoder: Maps candidate coordinates (3D) to latent space
        self.candidate_proj = nn.Sequential(
            nn.Linear(3, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Joint interaction layer
        self.joint_fc = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 1)
        )
        
        # Continuous Local 3D Refinement Offset Layer (SOTA Anchor Refinement)
        self.offset_fc = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 3)
        )
        
    def forward(self, history_df, context_features, candidates):
        # history_df: (batch_size, 10, 3)
        # context_features: (batch_size, 38)
        # candidates: (batch_size, 36, 3)
        batch_size = history_df.shape[0]
        
        # 1. Temporal encoding
        _, h_n = self.temporal_gru(history_df) # h_n: (1, batch_size, latent_dim)
        h_temp = h_n.squeeze(0) # (batch_size, latent_dim)
        
        # 2. Context fusion
        h_context = self.feature_proj(context_features)
        h_joint_context = h_temp + h_context # (batch_size, latent_dim)
        
        # 3. Spatial encoding for candidates
        cand_flat = candidates.view(-1, 3)
        h_cand_flat = self.candidate_proj(cand_flat) # (batch_size * 36, latent_dim)
        h_cand = h_cand_flat.view(batch_size, self.num_candidates, self.latent_dim) # (batch_size, 36, latent_dim)
        
        # 4. Compute Attention Scores & Local Offsets
        h_ctx_expanded = h_joint_context.unsqueeze(1).expand(-1, self.num_candidates, -1)
        fusion = torch.cat([h_ctx_expanded, h_cand], dim=-1) # (batch_size, 36, latent_dim * 2)
        
        scores = self.joint_fc(fusion).squeeze(-1)
        probs = torch.softmax(scores, dim=-1)
        
        # Predict local 3D offset refinements for each candidate: (batch_size, 36, 3)
        offsets = self.offset_fc(fusion)
        # Bound offsets to max 3.0cm to prevent wild divergence
        offsets = torch.tanh(offsets) * 0.03 
        
        return probs, offsets


class DifferentiableJointSelector(nn.Module):
    def __init__(self, feature_dim=38, latent_dim=64, num_candidates=36):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            ResBlock(latent_dim)
        )
        self.attention_selector = SpatioTemporalAttention(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            num_candidates=num_candidates
        )
        
    def forward(self, features, df, candidates):
        # Predict Selector Probabilities and Local Refinements
        probs, offsets = self.attention_selector(df, features, candidates) # probs: (N, 36), offsets: (N, 36, 3)
        
        # Blended coordinates with continuous offsets: sum(probs_i * (candidates_i + offset_i))
        refined_candidates = candidates + offsets
        pred_coords = torch.sum(probs.unsqueeze(-1) * refined_candidates, dim=1) # (batch_size, 3)
        return pred_coords, probs



class FocalSoftHitLoss(nn.Module):
    def __init__(self, delta=0.001, alpha=400.0, target_dist=0.01):
        super().__init__()
        self.delta = delta
        self.alpha = alpha
        self.target_dist = target_dist
        
    def forward(self, pred, target):
        d = torch.norm(pred - target, dim=1)
        
        # 1. Sigmoid-based Soft Hit loss targeting 1.0cm boundary
        soft_hit = 1 - torch.sigmoid(-(d - self.target_dist) * self.alpha)
        
        # 2. Near-miss penalty weight for 1.0cm < d <= 1.5cm
        focal_weight = torch.where((d > self.target_dist) & (d <= self.target_dist + 0.005), 2.5, 1.0)
        weighted_soft_hit = (focal_weight * soft_hit).mean()
        
        # 3. Huber Loss for continuous coordinate convergence
        huber = F_huber_loss(pred, target, self.delta)
        
        # Combined Loss
        return weighted_soft_hit + 120.0 * huber


def F_huber_loss(pred, target, delta=0.001):
    diff = pred - target
    abs_diff = torch.abs(diff)
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = abs_diff - quadratic
    loss = 0.5 * (quadratic ** 2) + delta * linear
    return loss.sum(dim=1).mean()

