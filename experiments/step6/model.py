import torch
import torch.nn as nn
import math
from .config import *

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

class TriAxisDiscreteTransformer(nn.Module):
    def __init__(self, input_size=12, d_model=256, nhead=8, num_layers=4, num_bins=21, dropout=0.1):
        super(TriAxisDiscreteTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Classification Heads (X, Y, Z)
        self.x_bin_head = nn.Linear(d_model, num_bins)
        self.y_bin_head = nn.Linear(d_model, num_bins)
        self.z_bin_head = nn.Linear(d_model, num_bins)
        
        # Offset Regression Heads (X, Y, Z)
        self.x_off_head = nn.Linear(d_model, 1)
        self.y_off_head = nn.Linear(d_model, 1)
        self.z_off_head = nn.Linear(d_model, 1)

    def forward(self, x, cv_prior, last_pos, rot_mat):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        feat = self.transformer_encoder(x)
        last_feat = feat[:, -1, :]
        
        # 1. Bins
        x_logits = self.x_bin_head(last_feat)
        y_logits = self.y_bin_head(last_feat)
        z_logits = self.z_bin_head(last_feat)
        
        # 2. Offsets
        x_off = self.x_off_head(last_feat)
        y_off = self.y_off_head(last_feat)
        z_off = self.z_off_head(last_feat)
        
        # Reconstruct (using most likely bins for monitoring)
        x_idx = torch.argmax(x_logits, dim=1)
        y_idx = torch.argmax(y_logits, dim=1)
        z_idx = torch.argmax(z_logits, dim=1)
        
        # Residual = (bin_idx - CENTER) * BIN_SIZE + offset * BIN_SIZE
        # offset was scaled to -0.5~0.5 of BIN_SIZE in dataset, but here let's just use it
        res_x = (x_idx.float() - CENTER_BIN) * BIN_SIZE + x_off.squeeze(-1) * BIN_SIZE
        res_y = (y_idx.float() - CENTER_BIN) * BIN_SIZE + y_off.squeeze(-1) * BIN_SIZE
        res_z = (z_idx.float() - CENTER_BIN) * BIN_SIZE + z_off.squeeze(-1) * BIN_SIZE
        
        pred_residual_m = torch.stack([res_x, res_y, res_z], dim=1)
        
        local_pred_rel = cv_prior + pred_residual_m
        global_pred_rel = torch.bmm(rot_mat.transpose(1, 2), local_pred_rel.unsqueeze(-1)).squeeze(-1)
        final_pos = last_pos + global_pred_rel
        
        return final_pos, (x_logits, y_logits, z_logits), (x_off, y_off, z_off)
