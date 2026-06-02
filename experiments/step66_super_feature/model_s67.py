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

class CrossAttentionVectorFieldNet(nn.Module):
    def __init__(self, context_dim=128, time_dim=32, num_heads=4, hidden_dim=128, max_norm=0.05):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.max_norm = max_norm
        self.x_proj = nn.Linear(3, hidden_dim)
        self.t_proj = nn.Linear(time_dim, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3)
        )
        
    def forward(self, x_t, t, h_context):
        t_emb = self.time_embed(t)
        q = (self.x_proj(x_t) + self.t_proj(t_emb)).unsqueeze(1) # [B, 1, H]
        
        k = h_context # [B, 10, H]
        v = h_context # [B, 10, H]
        
        attn_out, _ = self.attn(q, k, v) # [B, 1, H]
        v_t = self.out(attn_out.squeeze(1)) # [B, 3]
        
        v_norm = torch.norm(v_t, dim=-1, keepdim=True)
        scale = torch.where(v_norm > self.max_norm, self.max_norm / (v_norm + BaseEPS), torch.ones_like(v_norm))
        return v_t * scale

class SotaSpatiotemporalModel(nn.Module):
    def __init__(self, feature_dim=47, latent_dim=128, num_candidates=36, max_norm=0.05, d_mamba_in=9, dropout=0.0):
        super().__init__()
        self.num_candidates = num_candidates
        self.latent_dim = latent_dim
        self.max_norm = max_norm
        self.dropout = nn.Dropout(dropout)
        
        self.mamba_encoder = NativeMambaBlock(d_model=d_mamba_in, d_state=16, expand=4)
        self.mamba_proj = nn.Sequential(
            nn.Linear(d_mamba_in, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            self.dropout
        )
        self.context_proj = nn.Sequential(
            nn.Linear(feature_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            self.dropout
        )
        
        # GMM Regime Embeddings
        self.regime_embed = nn.Embedding(3, 16)
        self.regime_proj = nn.Sequential(
            nn.Linear(16, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            self.dropout
        )
        
        self.rel_embed = nn.Sequential(
            nn.Linear(11, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            self.dropout
        )
        self.clifford_embed = CliffordLinear(in_features=1, out_features=latent_dim // 8)
        self.cand_fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            self.dropout
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
        self.vector_field = CrossAttentionVectorFieldNet(context_dim=latent_dim, time_dim=32, num_heads=4, hidden_dim=latent_dim, max_norm=max_norm)
        
    def get_anchor(self, X, features, df_cartesian, df_spherical, candidates, regime_id):
        batch_size = X.shape[0]
        a_sph_diff = df_spherical[:, 1:] - df_spherical[:, :-1]
        a_sph_padded = torch.cat([torch.zeros(batch_size, 1, 3, device=X.device), a_sph_diff], dim=1)
        df_seq = torch.cat([df_cartesian, df_spherical, a_sph_padded], dim=-1)
        h_mamba_seq = self.mamba_encoder(df_seq)
        
        # Sequence-level context for vector field
        h_mamba_seq_proj = self.mamba_proj(h_mamba_seq) # [B, 10, H]
        h_ctx = self.context_proj(features) # [B, H]
        h_reg = self.regime_proj(self.regime_embed(regime_id)) # [B, H]
        h_context_seq = h_mamba_seq_proj + h_ctx.unsqueeze(1) + h_reg.unsqueeze(1) # [B, 10, H]
        
        # Last-frame context for anchor choice
        h_joint_ctx = h_context_seq[:, -1]
        
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
        return x_anchor, h_context_seq
        
    def forward_flow(self, X, features, df_cartesian, df_spherical, candidates, y, regime_id, noise_scale=0.01):
        x_anchor, h_context = self.get_anchor(X, features, df_cartesian, df_spherical, candidates, regime_id)
        noise = torch.randn_like(x_anchor) * noise_scale
        x_0 = x_anchor + noise
        t = torch.rand(X.shape[0], device=X.device)
        t_expand = t.unsqueeze(-1)
        x_t = (1 - t_expand) * x_0 + t_expand * y
        pred_v = self.vector_field(x_t, t, h_context)
        target_v = y - x_0
        return pred_v, target_v, x_0, h_context
        
    def predict(self, X, features, df_cartesian, df_spherical, candidates, regime_id, steps=1):
        x_anchor, h_context = self.get_anchor(X, features, df_cartesian, df_spherical, candidates, regime_id)
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

# The feature extraction and Focal loss classes are imported from step65 directly to reduce code clutter.
