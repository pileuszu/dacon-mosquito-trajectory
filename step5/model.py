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
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_node_nf * 2 + in_edge_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_node_nf + hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf)
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1, bias=False)
        )

    def forward(self, h, x, edge_index):
        row, col = edge_index
        dist = torch.sum((x[:, row] - x[:, col])**2, dim=-1, keepdim=True)
        edge_input = torch.cat([h[:, row], h[:, col], dist], dim=-1)
        m_ij = self.edge_mlp(edge_input)
        
        m_i = torch.zeros(h.size(0), h.size(1), self.hidden_nf, device=h.device)
        m_i.index_add_(1, row, m_ij)
        
        h_new = self.node_mlp(torch.cat([h, m_i], dim=-1))
        h = h + h_new
        
        w_ij = self.coord_mlp(m_ij)
        diff = x[:, row] - x[:, col]
        coord_update = torch.zeros_like(x)
        coord_update.index_add_(1, row, diff * w_ij)
        x = x + coord_update
        
        return h, x

class SpectralEqMotion(nn.Module):
    def __init__(self, seq_len=11, fft_dim=36, in_node_nf=3, hidden_nf=128, n_layers=4):
        super(SpectralEqMotion, self).__init__()
        self.seq_len = seq_len
        self.hidden_nf = hidden_nf
        
        # Spatial embedding
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        
        # Spectral embedding (FFT)
        self.fft_embedding = nn.Sequential(
            nn.Linear(fft_dim, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf)
        )
        
        # Layers
        self.layers = nn.ModuleList([
            EqMotionLayer(hidden_nf, 1, hidden_nf) for _ in range(n_layers)
        ])
        
        # Fully connected edges
        rows, cols = [], []
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        self.edge_index = torch.tensor([rows, cols], dtype=torch.long)

        # Output head: Predict residual correction
        self.predict_head = nn.Sequential(
            nn.Linear(hidden_nf * seq_len + hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 3)
        )

    def forward(self, x, fft):
        # x: (batch, seq_len, 3)
        # fft: (batch, fft_dim)
        batch_size = x.size(0)
        
        # Spatial initial state
        h = self.embedding(x)
        
        # Spectral initial state
        h_fft = self.fft_embedding(fft) # (batch, hidden_nf)
        
        # Inject spectral info into node features?
        # Let's add it to each node for global context
        h = h + h_fft.unsqueeze(1)
        
        edge_index = self.edge_index.to(x.device)
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
            
        # Combine spatial features and spectral context for final prediction
        h_flat = h.view(batch_size, -1)
        combined = torch.cat([h_flat, h_fft], dim=-1)
        
        residual = self.predict_head(combined)
        return residual

if __name__ == "__main__":
    # fft_dim calculation: 
    # rfft of 11 points -> 6 complex numbers. 
    # Magnitude (6) + Phase (6) per axis -> 12 * 3 = 36 features.
    model = SpectralEqMotion(fft_dim=36)
    x = torch.randn(8, 11, 3)
    fft = torch.randn(8, 36)
    out = model(x, fft)
    print(f"Output shape: {out.shape}")
