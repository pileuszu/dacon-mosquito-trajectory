import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# EPS for stability
BaseEPS = 1e-8

# =====================================================================
# 1. PURE PYTORCH MAMBA (SELECTIVE STATE SPACE MODEL) BLOCK
# =====================================================================
class NativeMambaBlock(nn.Module):
    """
    Windows-compatible pure PyTorch selective scan SSM block.
    Implements:
       h_t = A_bar * h_{t-1} + B_bar * x_t
       y_t = C * h_t + D * x_t
    with dynamic selection parameters Delta, B, C dependent on input x_t.
    """
    def __init__(self, d_model=64, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = d_model * expand
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # 1D Casual Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        # Parameter projections for selective SSM (Delta, B, C)
        self.x_proj = nn.Linear(self.d_inner, 1 + d_state * 2) # Delta_raw (1) + B (d_state) + C (d_state)
        
        # Learned parameter matrices A and D
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Project inputs
        projected = self.in_proj(x) # (batch, seq_len, d_inner * 2)
        x_branch, res_branch = torch.chunk(projected, 2, dim=-1)
        
        # Apply 1D Casual Conv
        x_branch = x_branch.transpose(1, 2) # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_branch)[:, :, :seq_len] # (batch, d_inner, seq_len)
        x_conv = x_conv.transpose(1, 2) # (batch, seq_len, d_inner)
        x_conv = F.silu(x_conv)
        
        # Selective parameter projection
        temp_proj = self.x_proj(x_conv) # (batch, seq, 1 + 2*d_state)
        Delta_raw, B, C = torch.split(temp_proj, [1, self.d_state, self.d_state], dim=-1)
        
        # Discretize parameters
        Delta = F.softplus(Delta_raw) # (batch, seq, 1)
        A = -torch.exp(self.A_log) # (d_inner, d_state)
        
        # Discretized A_bar: exp(Delta * A) -> shape: (batch, seq, d_inner, d_state)
        A_bar = torch.exp(Delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        
        # Discretized B_bar: Delta * B -> shape: (batch, seq, d_inner, d_state)
        B_bar = Delta.unsqueeze(-1) * B.unsqueeze(2)
        
        # 3. SELECTIVE SCAN: Compute hidden states h_t sequentially
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device)
        y_list = []
        
        for t in range(seq_len):
            x_t = x_conv[:, t].unsqueeze(-1) # (batch, d_inner, 1)
            h = A_bar[:, t] * h + B_bar[:, t] * x_t # (batch, d_inner, d_state)
            
            C_t = C[:, t].unsqueeze(1)
            y_t = torch.sum(C_t * h, dim=-1) # (batch, d_inner)
            y_list.append(y_t)
            
        y_ssm = torch.stack(y_list, dim=1) # (batch, seq, d_inner)
        
        # Apply multiplicative gating with residual branch
        y_gated = y_ssm * F.silu(res_branch) # (batch, seq, d_inner)
        
        # Output projection
        return self.out_proj(y_gated)


# =====================================================================
# 2. CLIFFORD (GEOMETRIC) ALGEBRA $Cl(3,0)$ NATIVE LINEAR LAYER
# =====================================================================
class CliffordLinear(nn.Module):
    """
    Geometric Algebra Cl(3,0) Equivariant Linear Layer.
    Embeds 3D vector variables into 8D Clifford Multivectors:
       [s, vx, vy, vz, bxy, byz, bzx, txyz]
    And computes geometric tensor product contraction natively.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_scalar = nn.Parameter(torch.randn(out_features, in_features) / np.sqrt(in_features))
        self.weight_vector = nn.Parameter(torch.randn(out_features, in_features) / np.sqrt(in_features))
        self.bias = nn.Parameter(torch.zeros(out_features, 8)) # bias for all 8 grades
        
    def forward(self, mv_x):
        # mv_x shape: (batch_size, in_features, 8)
        mv_even = mv_x[:, :, [0, 4, 5, 6]] # (batch, in, 4)
        mv_odd = mv_x[:, :, [1, 2, 3, 7]]  # (batch, in, 4)
        
        out_even = torch.einsum('o i, b i g -> b o g', self.weight_scalar, mv_even) # (batch, out, 4)
        out_odd = torch.einsum('o i, b i g -> b o g', self.weight_vector, mv_odd)   # (batch, out, 4)
        
        out_mv = torch.zeros(mv_x.shape[0], self.out_features, 8, device=mv_x.device)
        out_mv[:, :, [0, 4, 5, 6]] = out_even
        out_mv[:, :, [1, 2, 3, 7]] = out_odd
        
        return out_mv + self.bias.unsqueeze(0)


# =====================================================================
# 3. SINUSOIDAL TIME EMBEDDING
# =====================================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        # t: (batch_size,)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t * embeddings.unsqueeze(0) # (batch, half_dim)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# =====================================================================
# 4. CONTINUOUS VECTOR FIELD NETWORK
# =====================================================================
class VectorFieldNet(nn.Module):
    def __init__(self, context_dim=64, time_dim=32, hidden_dim=128):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        
        # Spatial coordinate projection
        self.x_proj = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.GELU()
        )
        
        # Fusion projection
        self.fc1 = nn.Linear(hidden_dim // 2 + time_dim + context_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.out = nn.Linear(hidden_dim, 3)
        
    def forward(self, x_t, t, h_context):
        # x_t: (batch, 3)
        # t: (batch,)
        # h_context: (batch, context_dim)
        t_emb = self.time_embed(t) # (batch, time_dim)
        x_emb = self.x_proj(x_t)   # (batch, hidden_dim // 2)
        
        fusion = torch.cat([x_emb, t_emb, h_context], dim=-1) # (batch, hidden + time + context)
        h = F.gelu(self.ln1(self.fc1(fusion)))
        h = F.gelu(self.ln2(self.fc2(h)))
        v_t = self.out(h) # (batch, 3)
        return v_t


# =====================================================================
# 5. SOTA FLOWS CONDITIONAL FLOW MATCHING (CFM) MODEL
# =====================================================================
class SotaCFMTrajectoryModel(nn.Module):
    def __init__(self, feature_dim=38, latent_dim=64, num_candidates=36):
        super().__init__()
        self.num_candidates = num_candidates
        self.latent_dim = latent_dim
        
        # 1. Temporal Encoder (Mamba)
        self.mamba_encoder = NativeMambaBlock(d_model=3, d_state=16, expand=4)
        self.mamba_proj = nn.Sequential(
            nn.Linear(3, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        
        # 2. Context embedding
        self.context_proj = nn.Sequential(
            nn.Linear(feature_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        
        # 3. Clifford Embeddings for Candidates
        self.clifford_embed = CliffordLinear(in_features=1, out_features=latent_dim // 8)
        self.cand_fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU()
        )
        
        # 4. Joint Attention over 36 continuous Anchors
        self.joint_attention = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 1)
        )
        
        # 5. Local 3D Refinement Offset
        self.offset_fc = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 3)
        )
        
        # 6. Vector Field Net for Flow Matching
        self.vector_field = VectorFieldNet(context_dim=latent_dim, time_dim=32, hidden_dim=128)
        
    def get_anchor(self, features, df, candidates):
        batch_size = df.shape[0]
        h_mamba_seq = self.mamba_encoder(df)
        h_temp = self.mamba_proj(h_mamba_seq[:, -1])
        h_ctx = self.context_proj(features)
        h_joint = h_temp + h_ctx # Context embedding
        
        # Convert candidates to grade-1 multivectors
        cand_mv = torch.zeros(batch_size * self.num_candidates, 1, 8, device=candidates.device)
        cand_mv[:, 0, 1:4] = candidates.view(-1, 3)
        h_cand_mv = self.clifford_embed(cand_mv)
        h_cand_flat = h_cand_mv.view(batch_size, self.num_candidates, -1)
        h_cand = self.cand_fc(h_cand_flat)
        
        h_joint_expanded = h_joint.unsqueeze(1).expand(-1, self.num_candidates, -1)
        fusion = torch.cat([h_joint_expanded, h_cand], dim=-1)
        
        scores = self.joint_attention(fusion).squeeze(-1)
        probs = torch.softmax(scores, dim=-1)
        
        offsets = self.offset_fc(fusion)
        offsets = torch.tanh(offsets) * 0.03 # 3cm boundary
        refined_candidates = candidates + offsets
        
        x_anchor = torch.sum(probs.unsqueeze(-1) * refined_candidates, dim=1)
        return x_anchor, h_joint
        
    def forward_flow(self, features, df, candidates, y, noise_scale=0.01):
        # y: target (batch, 3)
        batch_size = df.shape[0]
        
        x_anchor, h_context = self.get_anchor(features, df, candidates)
        
        # 1. Base point x_0: Add local noise perturbation to prior anchor
        noise = torch.randn_like(x_anchor) * noise_scale
        x_0 = x_anchor + noise
        
        # 2. Sample time t ~ Uniform(0, 1)
        t = torch.rand(batch_size, device=df.device)
        
        # 3. Flow interpolation x_t = (1 - t) * x_0 + t * y
        t_expand = t.unsqueeze(-1)
        x_t = (1 - t_expand) * x_0 + t_expand * y
        
        # 4. Predict velocity
        pred_v = self.vector_field(x_t, t, h_context)
        
        # Target velocity (y - x_0)
        target_v = y - x_0
        
        return pred_v, target_v, x_0, h_context
        
    def predict(self, features, df, candidates, steps=1):
        x_anchor, h_context = self.get_anchor(features, df, candidates)
        x = x_anchor # Inference start point (noise-free anchor)
        
        if steps == 1:
            # 1-step prediction (fast inference)
            t = torch.zeros(x.shape[0], device=df.device)
            v = self.vector_field(x, t, h_context)
            pred = x + v * 1.0
        elif steps == 2:
            # 2-step Euler prediction (consistency check)
            dt = 0.5
            # step 1: t=0
            t0 = torch.zeros(x.shape[0], device=df.device)
            v0 = self.vector_field(x, t0, h_context)
            x_half = x + v0 * dt
            # step 2: t=0.5
            t1 = torch.ones(x.shape[0], device=df.device) * 0.5
            v1 = self.vector_field(x_half, t1, h_context)
            pred = x_half + v1 * dt
        else:
            # N-step integration
            dt = 1.0 / steps
            for i in range(steps):
                t_val = i * dt
                t = torch.ones(x.shape[0], device=df.device) * t_val
                v = self.vector_field(x, t, h_context)
                x = x + v * dt
            pred = x
            
        return pred


# =====================================================================
# 6. 38D CLIFFORD MULTIVECTOR FEATURE EXTRACTION
# =====================================================================
def extract_features(X, mean_stats=None, std_stats=None):
    batch_size = X.shape[0]
    
    v = (X[:, 1:] - X[:, :-1]) / 0.04  # (batch_size, 10, 3)
    a = (v[:, 1:] - v[:, :-1]) / 0.04  # (batch_size, 9, 3)
    j = (a[:, 1:] - a[:, :-1]) / 0.04  # (batch_size, 8, 3)
    
    speeds = torch.norm(v, dim=2)
    acc_norms = torch.norm(a, dim=2)
    jerk_norms = torch.norm(j, dim=2)
    
    f1 = speeds[:, -1:]
    f2 = acc_norms[:, -1:]
    f3 = jerk_norms[:, -1:]
    
    f4 = speeds.mean(dim=1, keepdim=True)
    f5 = speeds.std(dim=1, keepdim=True)
    f6 = speeds.max(dim=1, keepdim=True).values
    
    f7 = acc_norms.mean(dim=1, keepdim=True)
    f8 = acc_norms.std(dim=1, keepdim=True)
    f9 = acc_norms.max(dim=1, keepdim=True).values
    
    f10 = jerk_norms.mean(dim=1, keepdim=True)
    f11 = jerk_norms.std(dim=1, keepdim=True)
    f12 = jerk_norms.max(dim=1, keepdim=True).values
    
    pos_std = X.std(dim=1)
    f13 = pos_std[:, 0:1]
    f14 = pos_std[:, 1:2]
    f15 = pos_std[:, 2:3]
    
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
    
    f29 = torch.sum(last_j * t, dim=1, keepdim=True) # J_T
    f30 = torch.sum(last_j * n, dim=1, keepdim=True) # J_N
    f31 = torch.sum(last_j * b, dim=1, keepdim=True) # J_B
    
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
    
    features = torch.cat([
        f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12,
        f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24,
        f25, f26, f27, f28, f29, f30, f31, f32, f33, f34, f35, f36, f37, f38
    ], dim=1)
    
    if mean_stats is None or std_stats is None:
        mean_stats = features.mean(dim=0)
        std_stats = features.std(dim=0)
        std_stats = torch.where(std_stats < 1e-6, torch.ones_like(std_stats), std_stats)
        
    features_scaled = (features - mean_stats) / std_stats
    
    mv_history = torch.zeros(batch_size, 10, 8, device=X.device)
    mv_history[:, :, 1:4] = v # velocity vectors
    
    df = X[:, 1:] - X[:, :-1]
    
    return features_scaled, df, mv_history, mean_stats, std_stats


# =====================================================================
# 7. SOTA FOCAL LOSS FUNCTION
# =====================================================================
class SotaFocalSoftHitLoss(nn.Module):
    def __init__(self, delta=0.001026, alpha=400.0, target_dist=0.01):
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
        
        # 3. Huber Loss
        diff = pred - target
        abs_diff = torch.abs(diff)
        quadratic = torch.clamp(abs_diff, max=self.delta)
        linear = abs_diff - quadratic
        huber = 0.5 * (quadratic ** 2) + self.delta * linear
        huber_loss = huber.sum(dim=1).mean()
        
        return weighted_soft_hit + 120.0 * huber_loss
