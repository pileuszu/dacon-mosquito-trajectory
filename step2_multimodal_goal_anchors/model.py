import torch
import torch.nn as nn
import pickle
import os

class MultimodalAnchorModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=2, n_anchors=6):
        super(MultimodalAnchorModel, self).__init__()
        self.n_anchors = n_anchors
        
        # Encoder (GRU)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        
        # 1. Classification Head: Predict which anchor is most likely
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_anchors)
        )
        
        # 2. Regression Head: Predict Offset (dx, dy, dz) for EACH anchor
        # Total output: n_anchors * 3
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_anchors * 3)
        )
        
    def forward(self, x):
        # x: [Batch, 11, 3]
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        
        # Probabilities for each anchor
        logits = self.cls_head(last_hidden) # [Batch, n_anchors]
        
        # Offsets for each anchor
        offsets = self.reg_head(last_hidden) # [Batch, n_anchors * 3]
        offsets = offsets.view(-1, self.n_anchors, 3) # [Batch, n_anchors, 3]
        
        return logits, offsets

def load_anchors(path='step2_multimodal_goal_anchors/anchors.pkl'):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return torch.tensor(pickle.load(f), dtype=torch.float32)
    else:
        # Fallback to zeros if not found yet
        return torch.zeros((6, 3))

if __name__ == "__main__":
    model = MultimodalAnchorModel()
    dummy_input = torch.randn(8, 11, 3)
    logits, offsets = model(dummy_input)
    print(f"Logits shape: {logits.shape}")   # [8, 6]
    print(f"Offsets shape: {offsets.shape}") # [8, 6, 3]
