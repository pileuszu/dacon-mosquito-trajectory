import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BaseEPS = 1e-8

class NativeMambaBlock(nn.Module):
    def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        self.x_proj = nn.Linear(self.d_inner, 1 + d_state * 2)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        projected = self.in_proj(x)
        x_branch, res_branch = torch.chunk(projected, 2, dim=-1)
        x_branch = x_branch.transpose(1, 2)
        x_conv = self.conv1d(x_branch)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        temp_proj = self.x_proj(x_conv)
        Delta_raw, B, C = torch.split(temp_proj, [1, self.d_state, self.d_state], dim=-1)
        Delta = F.softplus(Delta_raw)
        A = -torch.exp(self.A_log)
        A_bar = torch.exp(Delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        B_bar = Delta.unsqueeze(-1) * B.unsqueeze(2)
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device)
        y_list = []
        for t in range(seq_len):
            x_t = x_conv[:, t].unsqueeze(-1)
            h = A_bar[:, t] * h + B_bar[:, t] * x_t
            C_t = C[:, t].unsqueeze(1)
            y_t = torch.sum(C_t * h, dim=-1)
            y_list.append(y_t)
        y_ssm = torch.stack(y_list, dim=1)
        y_gated = y_ssm * F.silu(res_branch)
        return self.out_proj(y_gated)

class CliffordLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_scalar = nn.Parameter(torch.randn(out_features, in_features) / np.sqrt(in_features))
        self.weight_vector = nn.Parameter(torch.randn(out_features, in_features) / np.sqrt(in_features))
        self.bias = nn.Parameter(torch.zeros(out_features, 8))
    def forward(self, mv_x):
        mv_even = mv_x[:, :, [0, 4, 5, 6]]
        mv_odd = mv_x[:, :, [1, 2, 3, 7]]
        out_even = torch.einsum('o i, b i g -> b o g', self.weight_scalar, mv_even)
        out_odd = torch.einsum('o i, b i g -> b o g', self.weight_vector, mv_odd)
        out_mv = torch.zeros(mv_x.shape[0], self.out_features, 8, device=mv_x.device)
        out_mv[:, :, [0, 4, 5, 6]] = out_even
        out_mv[:, :, [1, 2, 3, 7]] = out_odd
        return out_mv + self.bias.unsqueeze(0)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class TangentProjectedVectorFieldNet(nn.Module):
    def __init__(self, context_dim=64, time_dim=32, hidden_dim=128, max_norm=0.08):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.max_norm = max_norm
        self.x_proj = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.GELU()
        )
        self.fc1 = nn.Linear(hidden_dim // 2 + time_dim + context_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, 3)
    def forward(self, x_t, t, h_context):
        t_emb = self.time_embed(t)
        x_emb = self.x_proj(x_t)
        fusion = torch.cat([x_emb, t_emb, h_context], dim=-1)
        h = F.gelu(self.ln1(self.fc1(fusion)))
        h = F.gelu(self.ln2(self.fc2(h)))
        v_t = self.out(h)
        v_norm = torch.norm(v_t, dim=-1, keepdim=True)
        scale = torch.where(v_norm > self.max_norm, self.max_norm / (v_norm + BaseEPS), torch.ones_like(v_norm))
        return v_t * scale

class SotaSpatiotemporalModel(nn.Module):
    def __init__(self, feature_dim=47, latent_dim=128, num_candidates=36, max_norm=0.05, d_mamba_in=9):
        super().__init__()
        self.num_candidates = num_candidates
        self.latent_dim = latent_dim
        self.max_norm = max_norm
        
        self.mamba_encoder = NativeMambaBlock(d_model=d_mamba_in, d_state=16, expand=4)
        self.mamba_proj = nn.Sequential(
            nn.Linear(d_mamba_in, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        self.context_proj = nn.Sequential(
            nn.Linear(feature_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        self.rel_embed = nn.Sequential(
            nn.Linear(11, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        self.clifford_embed = CliffordLinear(in_features=1, out_features=latent_dim // 8)
        self.cand_fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU()
        )
        self.joint_attention = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 1)
        )
        self.offset_fc = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 3)
        )
        self.vector_field = TangentProjectedVectorFieldNet(context_dim=latent_dim, time_dim=32, hidden_dim=128, max_norm=max_norm)
        
    def get_anchor(self, X, features, df_cartesian, df_spherical, candidates):
        batch_size = X.shape[0]
        a_sph_diff = df_spherical[:, 1:] - df_spherical[:, :-1]
        a_sph_padded = torch.cat([torch.zeros(batch_size, 1, 3, device=X.device), a_sph_diff], dim=1)
        df_seq = torch.cat([df_cartesian, df_spherical, a_sph_padded], dim=-1)
        h_mamba_seq = self.mamba_encoder(df_seq)
        h_temp = self.mamba_proj(h_mamba_seq[:, -1])
        h_ctx = self.context_proj(features)
        h_joint_ctx = h_temp + h_ctx
        X_exp = X.unsqueeze(2).expand(-1, -1, self.num_candidates, -1)
        cand_exp = candidates.unsqueeze(1).expand(-1, 11, -1, -1)
        rel_matrix = torch.norm(cand_exp - X_exp, dim=-1)
        rel_matrix_t = rel_matrix.transpose(1, 2)
        h_rel = self.rel_embed(rel_matrix_t)
        cand_mv = torch.zeros(batch_size * self.num_candidates, 1, 8, device=candidates.device)
        cand_mv[:, 0, 1:4] = candidates.view(-1, 3)
        h_cand_mv = self.clifford_embed(cand_mv)
        h_cand_flat = h_cand_mv.view(batch_size, self.num_candidates, -1)
        h_cand = self.cand_fc(h_cand_flat)
        h_joint_expanded = h_joint_ctx.unsqueeze(1).expand(-1, self.num_candidates, -1)
        fusion = torch.cat([h_joint_expanded, h_cand, h_rel], dim=-1)
        scores = self.joint_attention(fusion).squeeze(-1)
        probs = torch.softmax(scores, dim=-1)
        offsets = self.offset_fc(fusion)
        offsets = torch.tanh(offsets) * 0.03
        refined_candidates = candidates + offsets
        x_anchor = torch.sum(probs.unsqueeze(-1) * refined_candidates, dim=1)
        return x_anchor, h_joint_ctx
        
    def forward_flow(self, X, features, df_cartesian, df_spherical, candidates, y, noise_scale=0.01):
        x_anchor, h_context = self.get_anchor(X, features, df_cartesian, df_spherical, candidates)
        noise = torch.randn_like(x_anchor) * noise_scale
        x_0 = x_anchor + noise
        t = torch.rand(X.shape[0], device=X.device)
        t_expand = t.unsqueeze(-1)
        x_t = (1 - t_expand) * x_0 + t_expand * y
        pred_v = self.vector_field(x_t, t, h_context)
        target_v = y - x_0
        return pred_v, target_v, x_0, h_context
        
    def predict(self, X, features, df_cartesian, df_spherical, candidates, steps=1):
        x_anchor, h_context = self.get_anchor(X, features, df_cartesian, df_spherical, candidates)
        x = x_anchor
        if steps == 1:
            t = torch.zeros(x.shape[0], device=X.device)
            v = self.vector_field(x, t, h_context)
            pred = x + v * 1.0
        elif steps == 2:
            dt = 0.5
            t0 = torch.zeros(x.shape[0], device=X.device)
            v0 = self.vector_field(x, t0, h_context)
            x_half = x + v0 * dt
            t1 = torch.ones(x.shape[0], device=X.device) * 0.5
            v1 = self.vector_field(x_half, t1, h_context)
            pred = x_half + v1 * dt
        else:
            dt = 1.0 / steps
            for i in range(steps):
                t_val = i * dt
                t = torch.ones(x.shape[0], device=X.device) * t_val
                v = self.vector_field(x, t, h_context)
                x = x + v * dt
            pred = x
        return pred

def extract_features(X, mean_stats=None, std_stats=None):
    batch_size = X.shape[0]
    v = (X[:, 1:] - X[:, :-1]) / 0.04
    a = (v[:, 1:] - v[:, :-1]) / 0.04
    j = (a[:, 1:] - a[:, :-1]) / 0.04
    speeds = torch.norm(v, dim=2)
    acc_norms = torch.norm(a, dim=2)
    jerk_norms = torch.norm(j, dim=2)
    f1, f2, f3 = speeds[:, -1:], acc_norms[:, -1:], jerk_norms[:, -1:]
    f4, f5, f6 = speeds.mean(dim=1, keepdim=True), speeds.std(dim=1, keepdim=True), speeds.max(dim=1, keepdim=True).values
    f7, f8, f9 = acc_norms.mean(dim=1, keepdim=True), acc_norms.std(dim=1, keepdim=True), acc_norms.max(dim=1, keepdim=True).values
    f10, f11, f12 = jerk_norms.mean(dim=1, keepdim=True), jerk_norms.std(dim=1, keepdim=True), jerk_norms.max(dim=1, keepdim=True).values
    pos_std = X.std(dim=1)
    f13, f14, f15 = pos_std[:, 0:1], pos_std[:, 1:2], pos_std[:, 2:3]
    last_v = v[:, -1]
    f16 = torch.atan2(last_v[:, 1], last_v[:, 0]).unsqueeze(-1)
    f17 = torch.atan2(last_v[:, 2], torch.norm(last_v[:, :2], dim=1) + BaseEPS).unsqueeze(-1)
    last_a = a[:, -1]
    cross_prod = torch.cross(last_v, last_a, dim=1)
    f18 = (torch.norm(cross_prod, dim=1) / (torch.norm(last_v, dim=1) ** 3 + BaseEPS)).unsqueeze(-1)
    f19 = torch.norm(X[:, -1] - X[:, 0], dim=1, keepdim=True)
    f20 = speeds[:, -2:-1]
    f21 = speeds[:, -3:-2]
    f22 = acc_norms[:, -2:-1]
    f23 = acc_norms[:, -3:-2]
    f24 = jerk_norms[:, -2:-1]
    z_coords = X[:, :, 2]
    v_z = v[:, :, 2]
    a_z = a[:, :, 2]
    f25 = z_coords.std(dim=1, keepdim=True)
    f26 = v_z.mean(dim=1, keepdim=True)
    f27 = a_z[:, -1:]
    last_j = j[:, -1]
    cross_norm_sq = torch.sum(cross_prod ** 2, dim=1, keepdim=True)
    triple_prod = torch.sum(cross_prod * last_j, dim=1, keepdim=True)
    f28 = triple_prod / (cross_norm_sq + BaseEPS)
    t = last_v / (torch.norm(last_v, dim=1, keepdim=True) + BaseEPS)
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
    f29 = torch.sum(last_j * t, dim=1, keepdim=True)
    f30 = torch.sum(last_j * n, dim=1, keepdim=True)
    f31 = torch.sum(last_j * b, dim=1, keepdim=True)
    cross_prod_seq = torch.cross(v[:, 1:-1], a[:, 1:], dim=2)
    cross_norm_sq_seq = torch.sum(cross_prod_seq ** 2, dim=2)
    triple_prod_seq = torch.sum(cross_prod_seq * j, dim=2)
    torsion_seq = triple_prod_seq / (cross_norm_sq_seq + BaseEPS)
    f32 = torsion_seq.std(dim=1, keepdim=True)
    curv_seq = torch.norm(cross_prod_seq, dim=2) / (torch.norm(v[:, 1:-1], dim=2) ** 3 + BaseEPS)
    f33 = curv_seq.std(dim=1, keepdim=True)
    omega_last = torch.norm(cross_prod, dim=1, keepdim=True) / (torch.norm(last_v, dim=1, keepdim=True) ** 2 + BaseEPS)
    f34 = omega_last
    prev_cross = torch.cross(v[:, -2], a[:, -2], dim=1)
    omega_prev = torch.norm(prev_cross, dim=1, keepdim=True) / (torch.norm(v[:, -2], dim=1, keepdim=True) ** 2 + BaseEPS)
    f35 = (omega_last - omega_prev) / 0.04
    f36 = speeds[:, -1:] / (speeds[:, -2:-1] + BaseEPS)
    cos_theta_a = torch.sum(a[:, -1] * a[:, -2], dim=1, keepdim=True) / (acc_norms[:, -1:] * acc_norms[:, -2:-1] + BaseEPS)
    cos_theta_a = torch.clamp(cos_theta_a, -1.0, 1.0)
    f37 = torch.acos(cos_theta_a)
    f38 = jerk_norms.std(dim=1, keepdim=True)
    
    rho_seq = torch.norm(X, dim=2)
    theta_seq = torch.atan2(X[:, :, 1], X[:, :, 0])
    phi_seq = torch.acos(torch.clamp(X[:, :, 2] / (rho_seq + BaseEPS), -1.0, 1.0))
    d_rho = (rho_seq[:, 1:] - rho_seq[:, :-1]) / 0.04
    d_theta = (theta_seq[:, 1:] - theta_seq[:, :-1]) / 0.04
    d_phi = (phi_seq[:, 1:] - phi_seq[:, :-1]) / 0.04
    d_theta = torch.remainder(d_theta + np.pi, 2 * np.pi) - np.pi
    d_phi = torch.remainder(d_phi + np.pi, 2 * np.pi) - np.pi
    
    df_spherical = torch.stack([d_rho, d_theta, d_phi], dim=-1)
    
    f39 = d_rho[:, -1:]
    f40 = d_theta[:, -1:]
    f41 = d_phi[:, -1:]
    f42 = d_rho.mean(dim=1, keepdim=True)
    f43 = d_theta.std(dim=1, keepdim=True)
    f44 = d_phi.std(dim=1, keepdim=True)
    
    a_spherical = df_spherical[:, 1:] - df_spherical[:, :-1]
    f45 = a_spherical[:, -1, 0:1]
    f46 = a_spherical[:, -1, 1:2]
    f47 = a_spherical[:, -1, 2:3]
    polaris_features = [f39, f40, f41, f42, f43, f44, f45, f46, f47]
    
    features = torch.cat(
        [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12,
         f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24,
         f25, f26, f27, f28, f29, f30, f31, f32, f33, f34, f35, f36, f37, f38] + polaris_features, dim=1
    )
    
    if mean_stats is None or std_stats is None:
        mean_stats = features.mean(dim=0)
        std_stats = features.std(dim=0)
        std_stats = torch.where(std_stats < 1e-6, torch.ones_like(std_stats), std_stats)
    features_scaled = (features - mean_stats) / std_stats
    df_cartesian = X[:, 1:] - X[:, :-1]
    return features_scaled, df_cartesian, df_spherical, mean_stats, std_stats

class SotaFocalSoftHitLoss(nn.Module):
    def __init__(self, delta=0.001026, alpha=400.0, target_dist=0.01):
        super().__init__()
        self.delta = delta
        self.alpha = alpha
        self.target_dist = target_dist
    def forward(self, pred, target):
        d = torch.norm(pred - target, dim=1)
        soft_hit = 1 - torch.sigmoid(-(d - self.target_dist) * self.alpha)
        focal_weight = torch.where((d > self.target_dist) & (d <= self.target_dist + 0.005), 2.5, 1.0)
        weighted_soft_hit = (focal_weight * soft_hit).mean()
        diff = pred - target
        abs_diff = torch.abs(diff)
        quadratic = torch.clamp(abs_diff, max=self.delta)
        linear = abs_diff - quadratic
        huber = 0.5 * (quadratic ** 2) + self.delta * linear
        huber_loss = huber.sum(dim=1).mean()
        return weighted_soft_hit + 120.0 * huber_loss
