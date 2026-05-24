import torch
import torch.nn as nn
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

class BiomechanicalTransformer(nn.Module):
    def __init__(self, input_size=12, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.15):
        super(BiomechanicalTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Multi-task head
        # 1. Residual Regression Head (cm scale)
        self.residual_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)
        )
        
        # 2. Maneuver State Classifier Head (Auxiliary)
        # Predicts if the movement is a 'Saccade' (high intensity maneuver)
        self.state_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Probability of being in 'Saccade' state
        )

    def forward(self, x, cv_prior, last_pos, rot_mat):
        """
        x: (batch, 10, 12)
        cv_prior: (batch, 3) - Rotated
        last_pos: (batch, 3) - Absolute
        rot_mat: (batch, 3, 3)
        """
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        # (batch, 10, d_model)
        feat = self.transformer_encoder(x)
        
        # Use last token for prediction
        last_feat = feat[:, -1, :]
        
        # 1. Scaled Residual (in cm)
        pred_residual_cm = self.residual_head(last_feat)
        
        # 2. Maneuver state
        state_logit = self.state_head(last_feat)
        
        # 3. Reconstruct Final Position
        # Final_pos = Last_pos + Rot_Mat_Inv @ (CV_Prior_Rot + Pred_Res_Rot / Scale)
        # Note: rot_mat aligns global to local. To go back, we use rot_mat.T
        
        pred_residual_m = pred_residual_cm / 100.0 # Convert back to meters for reconstruction
        
        # Apply inverse rotation to the local prediction
        # Local space: CV_Prior_Rot + Pred_Res_Rot_m
        local_pred_rel = cv_prior + pred_residual_m
        
        # Back to global space: rot_mat^T @ local_pred_rel
        # batch matrix multiplication: (B, 3, 3) @ (B, 3, 1)
        global_pred_rel = torch.bmm(rot_mat.transpose(1, 2), local_pred_rel.unsqueeze(-1)).squeeze(-1)
        
        final_pos = last_pos + global_pred_rel
        
        return final_pos, pred_residual_cm, state_logit
