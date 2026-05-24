import torch
import torch.nn as nn
import math

class CandidateSelector(nn.Module):
    def __init__(self, input_dim=12, num_candidates=27, d_model=256, dropout=0.1, cand_feat_dim=9):
        super().__init__()
        self.d_model = d_model
        self.num_candidates = num_candidates
        
        # Sequence Encoder: GRU is more efficient for short sequences
        self.input_proj = nn.Linear(input_dim, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        
        # Merge Bidirectional Context
        self.context_proj = nn.Linear(d_model * 2, d_model)
        
        # Candidate Feature Projector
        self.cand_proj = nn.Linear(cand_feat_dim, d_model)
        
        # Cross-Attention / Scoring Head
        self.selector = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        # Correction Head
        self.correction_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
            nn.Tanh()
        )

    def forward(self, seq, candidates, cand_feats, rot_mat):
        B, T, _ = seq.shape
        N = candidates.shape[1]
        
        # 1. Context Extraction (GRU)
        x = self.input_proj(seq) # (B, T, d_model)
        out, _ = self.gru(x) # (B, T, d_model * 2)
        ctx_last = self.context_proj(out[:, -1, :]) # (B, d_model)
        
        # 2. Candidate Matching
        cand_embed = self.cand_proj(cand_feats) # (B, N, d_model)
        ctx_expanded = ctx_last.unsqueeze(1).expand(-1, N, -1) # (B, N, d_model)
        
        combined = torch.cat([ctx_expanded, cand_embed], dim=-1) # (B, N, d_model * 2)
        logits = self.selector(combined).squeeze(-1) # (B, N)
        
        # 3. Correction
        local_corr = self.correction_head(ctx_last) * 0.01 
        global_corr = torch.bmm(rot_mat.transpose(-1, -2), local_corr.unsqueeze(-1)).squeeze(-1)
        
        return logits, global_corr

# Keep the old name as alias for compatibility
CandidateTransformerSelector = CandidateSelector
