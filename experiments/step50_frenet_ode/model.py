import torch
import torch.nn as nn
import numpy as np

# EPS for numerical stability
EPS = 1e-8

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
    f17 = torch.atan2(last_v[:, 2], torch.norm(last_v[:, :2], dim=1) + 1e-8).unsqueeze(-1)  # pitch angle
    
    last_a = a[:, -1]
    cross_prod = torch.cross(last_v, last_a, dim=1)
    # curvature calculation
    f18 = (torch.norm(cross_prod, dim=1) / (torch.norm(last_v, dim=1) ** 3 + 1e-8)).unsqueeze(-1)  # curvature
    
    f19 = torch.norm(X[:, -1] - X[:, 0], dim=1, keepdim=True)  # total displacement
    
    f20 = speeds[:, -2:-1]  # speed at t=9
    f21 = speeds[:, -3:-2]  # speed at t=8
    f22 = acc_norms[:, -2:-1]  # accel norm at t=8
    f23 = acc_norms[:, -3:-2]  # accel norm at t=7
    f24 = jerk_norms[:, -2:-1]  # jerk norm at t=7
    
    features = torch.cat([
        f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12,
        f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24
    ], dim=1)
    
    # 3. Compute scale stats if not provided
    if mean_stats is None or std_stats is None:
        mean_stats = features.mean(dim=0)
        std_stats = features.std(dim=0)
        std_stats[std_stats < 1e-6] = 1.0
        
    features_scaled = (features - mean_stats) / std_stats
    
    # 4. Compute initial R matrix (at t=0ms for history alignment)
    t = last_v / (torch.norm(last_v, dim=1, keepdim=True) + 1e-8)  # (tangent)
    
    acc_perp = last_a - torch.sum(last_a * t, dim=1, keepdim=True) * t
    n_norm = torch.norm(acc_perp, dim=1, keepdim=True)
    n = acc_perp / (n_norm + 1e-8)
    
    fallback = torch.zeros_like(n)
    axis = torch.argmin(torch.abs(t), dim=1)
    fallback[torch.arange(batch_size), axis] = 1.0
    fallback = fallback - torch.sum(fallback * t, dim=1, keepdim=True) * t
    fallback = fallback / (torch.norm(fallback, dim=1, keepdim=True) + 1e-8)
    n = torch.where(n_norm > 1e-6, n, fallback)
    
    b = torch.cross(t, n, dim=1)
    b = b / (torch.norm(b, dim=1, keepdim=True) + 1e-8)
    R = torch.stack([t, n, b], dim=2)
    
    # 5. Steering indicator classification (m/s and m/s^2 metrics)
    # is_steering = curvature > 6.0 OR acc_perp_norm > 1.8
    # note: last_a is in m/s^2
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
            nn.Dropout(0.15),
            nn.Linear(dim, dim)
        )
        self.ln = nn.LayerNorm(dim)
        
    def forward(self, x):
        return self.ln(x + self.net(x))


class FrenetAccelerationField(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        # Input: pos (3) + vel (3) + context (latent_dim) + Yaw (1) + reference speed (1)
        self.net = nn.Sequential(
            nn.Linear(3 + 3 + latent_dim + 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            ResBlock(64),
            nn.Linear(64, 3)
        )
        
    def forward(self, pos, vel, latent, theta, speed):
        if theta.dim() == 1:
            theta = theta.unsqueeze(-1)
        if speed.dim() == 1:
            speed = speed.unsqueeze(-1)
        inputs = torch.cat([pos, vel, latent, theta, speed], dim=-1)
        return self.net(inputs)


class FrenetNeuralODEModel(nn.Module):
    def __init__(self, input_dim=24, latent_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            ResBlock(latent_dim)
        )
        self.accel_field = FrenetAccelerationField(latent_dim=latent_dim)
        
        self.learned_damping = nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))
        self.local_bias = nn.Parameter(torch.zeros(3))
        self.global_bias = nn.Parameter(torch.zeros(3))
        
        self.dt_physical = 0.08  # RK4 integration interval: 80ms
        self._last_accels = []
        
    def _compute_frenet_frame(self, vel, accel):
        # vel: (batch_size, 3), accel: (batch_size, 3)
        batch_size = vel.shape[0]
        device = vel.device
        
        speeds = torch.norm(vel, dim=1, keepdim=True)
        t = vel / (speeds + 1e-8)
        
        acc_perp = accel - torch.sum(accel * t, dim=1, keepdim=True) * t
        n_norm = torch.norm(acc_perp, dim=1, keepdim=True)
        n = acc_perp / (n_norm + 1e-8)
        
        # Fallback in case of straight motion
        fallback = torch.zeros_like(n)
        axis = torch.argmin(torch.abs(t), dim=1)
        fallback[torch.arange(batch_size), axis] = 1.0
        fallback = fallback - torch.sum(fallback * t, dim=1, keepdim=True) * t
        fallback = fallback / (torch.norm(fallback, dim=1, keepdim=True) + 1e-8)
        
        n = torch.where(n_norm > 1e-6, n, fallback)
        
        b = torch.cross(t, n, dim=1)
        b = b / (torch.norm(b, dim=1, keepdim=True) + 1e-8)
        
        R_t = torch.stack([t, n, b], dim=2) # (batch_size, 3, 3)
        return R_t
        
    def _ode_derivative(self, pos, vel, latent, theta, speed, current_accel):
        # Compute Frenet frame R_t dynamically based on current vel & accel
        R_t = self._compute_frenet_frame(vel, current_accel)
        
        # Acceleration field outputs local acceleration in Frenet space
        a_local = self.accel_field(pos, vel, latent, theta, speed)
        
        # Rotate back to global space
        a_global = torch.einsum('nij,nj->ni', R_t, a_local)
        
        dpos = vel
        dvel = -self.learned_damping * vel + a_global
        return dpos, dvel, a_global
        
    def _rk4_simulation(self, init_pos, init_vel, latent, theta, speed, initial_accel):
        dt = self.dt_physical
        
        # Step 1: k1
        dp1, dv1, a1 = self._ode_derivative(init_pos, init_vel, latent, theta, speed, initial_accel)
        pos_k2 = init_pos + 0.5 * dt * dp1
        vel_k2 = init_vel + 0.5 * dt * dv1
        
        # Step 2: k2 (use k1 acceleration as pilot accel)
        dp2, dv2, a2 = self._ode_derivative(pos_k2, vel_k2, latent, theta, speed, a1)
        pos_k3 = init_pos + 0.5 * dt * dp2
        vel_k3 = init_vel + 0.5 * dt * dv2
        
        # Step 3: k3 (use k2 acceleration as pilot accel)
        dp3, dv3, a3 = self._ode_derivative(pos_k3, vel_k3, latent, theta, speed, a2)
        pos_k4 = init_pos + dt * dp3
        vel_k4 = init_vel + dt * dv3
        
        # Step 4: k4 (use k3 acceleration as pilot accel)
        dp4, dv4, a4 = self._ode_derivative(pos_k4, vel_k4, latent, theta, speed, a3)
        
        final_pos = init_pos + (dt / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4)
        final_vel = init_vel + (dt / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4)
        
        self._last_accels = [a1, a2, a3, a4]
        return final_pos, final_vel
        
    def forward(self, features, diffs, p_last, theta, speed, R, initial_accel):
        latent = self.backbone(features)
        
        # Convert historical differences to local frame for initializing velocity
        diffs_local = torch.matmul(diffs, R)
        
        init_pos = torch.zeros_like(p_last)
        # initial velocity in local frame (at t=0ms for solver initialization)
        init_vel = diffs_local[:, -1] / 0.04
        
        # Convert initial velocity back to global space for ODE solver
        init_vel_global = torch.einsum('nij,nj->ni', R, init_vel)
        
        # Run dynamic RK4 수치 적분
        pos, vel = self._rk4_simulation(init_pos, init_vel_global, latent, theta, speed, initial_accel)
        
        pred_local = pos + self.local_bias
        # final prediction is global coordinate projection
        pred_global = p_last + pred_local + self.global_bias
        return pred_global
