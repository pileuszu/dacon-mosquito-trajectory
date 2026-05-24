import torch
import torch.nn as nn

class LSTMHybridModel(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, num_layers=3):
        super(LSTMHybridModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer (handles x,y,z + v + a)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # Enhanced Regression head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # Output is Residual (dx, dy, dz)
        )

    def forward(self, x, cv_prior, last_pos):
        """
        x: (batch, seq_len, 9) - History features
        cv_prior: (batch, 3) - Constant Velocity prediction
        last_pos: (batch, 3) - Last known absolute coordinate
        """
        # LSTM feature extraction
        out, _ = self.lstm(x)
        
        # Context from the whole sequence (last hidden state)
        last_hidden = out[:, -1, :]
        
        # Predict correction (residual)
        residual = self.fc(last_hidden)
        
        # Final prediction = Last Point + CV Baseline + Model Correction
        final_pos = last_pos + cv_prior + residual
        
        return final_pos, residual
