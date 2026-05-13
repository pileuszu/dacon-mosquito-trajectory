import torch
import torch.nn as nn
import os
import pickle

class ArenaAwareAnchorModel(nn.Module):
    def __init__(self, input_dim=3, arena_dim=7, hidden_dim=128, num_layers=2, n_anchors=6):
        super(ArenaAwareAnchorModel, self).__init__()
        self.n_anchors = n_anchors
        
        # 1. Temporal Encoder (GRU)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        
        # 2. Arena Context Encoder (MLP)
        self.arena_mlp = nn.Sequential(
            nn.Linear(arena_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Combined Feature Dimension
        combined_dim = hidden_dim + 32
        
        # 3. Heads
        self.cls_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_anchors)
        )
        
        self.reg_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_anchors * 3)
        )
        
    def forward(self, x, arena_feat):
        # x: [Batch, 11, 3], arena_feat: [Batch, 7]
        
        # Encode Temporal
        out, _ = self.gru(x)
        temporal_emb = out[:, -1, :]
        
        # Encode Arena Context
        arena_emb = self.arena_mlp(arena_feat)
        
        # Concatenate
        combined = torch.cat([temporal_emb, arena_emb], dim=1)
        
        logits = self.cls_head(combined)
        offsets = self.reg_head(combined).view(-1, self.n_anchors, 3)
        
        return logits, offsets

def load_anchors(path='step2_multimodal_goal_anchors/anchors.pkl'):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return torch.tensor(pickle.load(f), dtype=torch.float32)
    else:
        # Canonical anchors if not found
        return torch.zeros((6, 3))
