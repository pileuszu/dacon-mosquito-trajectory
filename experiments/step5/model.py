import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiModalGMMTransformer(nn.Module):
    def __init__(self, input_size=12, d_model=256, nhead=8, num_layers=4, num_modes=6, dropout=0.1):
        super(MultiModalGMMTransformer, self).__init__()
        
        self.num_modes = num_modes
        self.d_model = d_model
        
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # GMM Head
        # For each mode: mu(3), log_sigma(3), pi(1)
        # Total output size: num_modes * (3 + 3 + 1)
        self.gmm_head = nn.Linear(d_model, num_modes * 7)

    def forward(self, x, cv_prior, last_pos, rot_mat):
        """
        x: (batch, 10, 12) - Rotated features
        """
        x = self.embedding(x)
        x = self.pos_encoder(x)
        feat = self.transformer_encoder(x)
        last_feat = feat[:, -1, :] # (batch, d_model)
        
        gmm_out = self.gmm_head(last_feat) # (batch, num_modes * 7)
        gmm_out = gmm_out.view(-1, self.num_modes, 7)
        
        # Split into mu, log_sigma, pi
        mu = gmm_out[:, :, 0:3]          # (batch, num_modes, 3)
        log_sigma = gmm_out[:, :, 3:6]   # (batch, num_modes, 3)
        pi_logits = gmm_out[:, :, 6]     # (batch, num_modes)
        
        # Softmax for mode probabilities
        pi = F.softmax(pi_logits, dim=1)
        
        # For evaluation/inference, we often want the most likely mode
        # or the weighted average. For NLL loss, we need all.
        
        # Reconstruct final position for the most likely mode (for monitoring)
        best_mode_idx = torch.argmax(pi, dim=1)
        best_mu = mu[torch.arange(mu.size(0)), best_mode_idx] # (batch, 3)
        
        # Back to global space (same logic as step4)
        pred_residual_m = best_mu / 100.0
        local_pred_rel = cv_prior + pred_residual_m
        global_pred_rel = torch.bmm(rot_mat.transpose(1, 2), local_pred_rel.unsqueeze(-1)).squeeze(-1)
        final_pos = last_pos + global_pred_rel
        
        return final_pos, mu, log_sigma, pi
