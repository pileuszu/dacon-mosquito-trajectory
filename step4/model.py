import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class EqMotionLayer(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, act_fn=nn.SiLU()):
        super(EqMotionLayer, self).__init__()
        self.hidden_nf = hidden_nf
        
        # Edge model
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_node_nf * 2 + in_edge_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )
        
        # Node model
        self.node_mlp = nn.Sequential(
            nn.Linear(in_node_nf + hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf)
        )
        
        # Equivariant coordinate update model
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1, bias=False)
        )

    def forward(self, h, x, edge_index, edge_attr=None):
        # h: node features (batch, seq_len, hidden_nf)
        # x: coordinates (batch, seq_len, 3)
        # edge_index: (2, num_edges)
        
        row, col = edge_index
        
        # Invariant feature: squared distance
        dist = torch.sum((x[:, row] - x[:, col])**2, dim=-1, keepdim=True) # (batch, num_edges, 1)
        
        # Edge features
        edge_input = torch.cat([h[:, row], h[:, col], dist], dim=-1)
        if edge_attr is not None:
            edge_input = torch.cat([edge_input, edge_attr], dim=-1)
            
        m_ij = self.edge_mlp(edge_input) # (batch, num_edges, hidden_nf)
        
        # Aggregate edge features to nodes
        m_i = torch.zeros(h.size(0), h.size(1), self.hidden_nf, device=h.device)
        m_i.index_add_(1, row, m_ij) # Sum over j
        
        # Update node features
        h_new = self.node_mlp(torch.cat([h, m_i], dim=-1))
        h = h + h_new
        
        # Update coordinates (Equivariant update)
        # delta_x = sum_{j} (x_i - x_j) * w_ij
        w_ij = self.coord_mlp(m_ij) # (batch, num_edges, 1)
        # Limit coordinate update magnitude for stability
        # w_ij = torch.tanh(w_ij) 
        
        diff = x[:, row] - x[:, col]
        coord_update = torch.zeros_like(x)
        coord_update.index_add_(1, row, diff * w_ij)
        
        x = x + coord_update
        
        return h, x

class EqMotion(nn.Module):
    def __init__(self, seq_len=11, in_node_nf=3, hidden_nf=64, n_layers=4):
        super(EqMotion, self).__init__()
        self.seq_len = seq_len
        self.hidden_nf = hidden_nf
        
        # Initial embedding
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        
        # Layers
        self.layers = nn.ModuleList([
            EqMotionLayer(hidden_nf, 1, hidden_nf) for _ in range(n_layers)
        ])
        
        # Fully connected edges (all-to-all for temporal sequence)
        # Or just sequential edges. For small sequence, all-to-all is fine.
        rows, cols = [], []
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        self.edge_index = torch.tensor([rows, cols], dtype=torch.long)

        # Output head: Predict 80ms pos (relative to 0ms pos)
        self.predict_head = nn.Sequential(
            nn.Linear(hidden_nf * seq_len, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 3)
        )

    def forward(self, x):
        # x: (batch, seq_len, 3)
        batch_size = x.size(0)
        
        # Preprocessing: Shift so that the last point (0ms) is at the origin?
        # No, EqMotion is equivariant, so it should handle absolute coords.
        # But for stability, we can use relative coords from the first point.
        origin = x[:, -1:, :] # Use 0ms as origin
        x_rel = x - origin
        
        # Initial node features (could be velocity or just initial embedding)
        h = self.embedding(x_rel) 
        
        edge_index = self.edge_index.to(x.device)
        
        for layer in self.layers:
            h, x_rel = layer(h, x_rel, edge_index)
            
        # Global pooling or flatten for prediction
        h_flat = h.view(batch_size, -1)
        
        # The coordinates x_rel are also updated equivariantly.
        # We can predict the final displacement from the last point's feature or updated coord.
        # Let's use the features to predict the final residual.
        delta_target = self.predict_head(h_flat)
        
        # Final prediction: 0ms pos + predicted displacement
        prediction = origin.squeeze(1) + delta_target
        
        return prediction

if __name__ == "__main__":
    model = EqMotion()
    dummy_input = torch.randn(8, 11, 3)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
