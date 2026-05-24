import torch
import torch.nn as nn

class LSTMResidualModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super(LSTMResidualModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Regression head for residual
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # Output is (dx, dy, dz)
        )

    def forward(self, x, cv_prior, last_pos):
        """
        x: (batch, seq_len, 3) - Normalized sequence (relative to last_pos)
        cv_prior: (batch, 3) - CV prediction (relative to last_pos)
        last_pos: (batch, 3) - Absolute coordinate of the last point
        """
        # LSTM feature extraction
        # out: (batch, seq_len, hidden_size)
        out, _ = self.lstm(x)
        
        # Take the last hidden state
        last_hidden = out[:, -1, :]
        
        # Predict residual relative to CV Prior
        residual = self.fc(last_hidden)
        
        # Final prediction (relative to last_pos)
        # Pred_rel = CV_Prior + Residual
        pred_rel = cv_prior + residual
        
        # Final absolute prediction
        final_pos = last_pos + pred_rel
        
        return final_pos, residual
