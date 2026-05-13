import torch
import torch.nn as nn

class BaselineGRU(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=2, output_dim=3):
        super(BaselineGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Simple GRU Encoder
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        
        # Fully connected head to predict the future relative displacement
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        # x: [Batch, Sequence_Length (11), 3]
        out, _ = self.gru(x)
        
        # Take the last hidden state of the sequence
        last_hidden = out[:, -1, :]
        
        prediction = self.fc(last_hidden)
        return prediction

if __name__ == "__main__":
    # Test model
    model = BaselineGRU()
    dummy_input = torch.randn(8, 11, 3)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Should be [8, 3]
