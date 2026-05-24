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

class TransformerResidualModel(nn.Module):
    def __init__(self, input_size=9, d_model=64, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1):
        super(TransformerResidualModel, self).__init__()
        
        # Projection layer to match d_model
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Regression head for residual
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # Output: (dx, dy, dz) residual relative to CV Prior
        )

    def forward(self, x, cv_prior, last_pos):
        """
        x: (batch, seq_len, 9) - History features
        cv_prior: (batch, 3) - CV prediction relative to last_pos
        last_pos: (batch, 3) - Last known absolute coordinate
        """
        # 1. Feature Embedding & Positional Encoding
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        # 2. Transformer Encoding
        # x: (batch, seq_len, d_model)
        out = self.transformer_encoder(x)
        
        # 3. Global feature (using last token or mean pooling)
        # Using the last token for sequence context
        last_token = out[:, -1, :]
        
        # 4. Predict Residual
        residual = self.fc(last_token)
        
        # 5. Final Prediction
        # Final_pos = Last_pos + CV_Prior + Residual
        final_pos = last_pos + cv_prior + residual
        
        return final_pos, residual
