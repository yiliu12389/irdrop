"""GNN model for IR drop hotspot prediction."""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from torchdiffeq import odeint

class PDNTelegraphODE(nn.Module):
    def __init__(self, edge_fdim=6, node_fdim=3, hidden=64):
        super().__init__()
        self.G_net = nn.Sequential(
            nn.Linear(edge_fdim, hidden), nn.SiLU(),
            nn.Linear(hidden, 1), nn.Softplus()
        )
        self.C_net = nn.Sequential(
            nn.Linear(node_fdim, hidden), nn.SiLU(),
            nn.Linear(hidden, 1), nn.Softplus()
        )

    def prepare(self, data, power_seq):
            src, dst = data.edge_index
            self.src, self.dst = src, dst
            self.N = data.x.shape[0]
            self.G = self.G_net(data.edge_attr).squeeze(-1)   
            self.C = self.C_net(data.x).squeeze(-1)           
            self.power_seq = power_seq   
            self.T = power_seq.shape[0]
            self.supply_mask = (data.x[:, 1] == 0) & \
                            (data.x[:, 0] > 0.85)  
    
    def forward(self, t, V):
        I_wire = self.G * (V[self.dst] - V[self.src])   # (E,)
        net_I  = scatter_add(I_wire,  self.dst, dim=0, dim_size=self.N) \
               - scatter_add(I_wire,  self.src, dim=0, dim_size=self.N)
        
        t_f    = t.item() * (self.T - 1)
        t0, t1 = int(t_f), min(int(t_f)+1, self.T-1)
        alpha  = t_f - t0
        I_inj  = (1-alpha)*self.power_seq[t0] + alpha*self.power_seq[t1]

        dVdt   = (net_I - I_inj) / self.C
        dVdt[self.supply_mask] = 0.0   
        return dVdt
        
class PairwiseRankingLoss(nn.Module):
    def __init__(self, n_pairs: int = 1024, top_frac: float = 0.10, bot_frac: float = 0.50):
        super().__init__()
        self.n_pairs  = n_pairs
        self.top_frac = top_frac
        self.bot_frac = bot_frac

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        N      = scores.size(0)
        device = scores.device

        sorted_idx = torch.argsort(labels)
        n_top = max(1, int(N * self.top_frac))
        n_bot = max(1, int(N * self.bot_frac))
        top_idx = sorted_idx[-n_top:]   # highest IR drop nodes
        bot_idx = sorted_idx[:n_bot]    # lowest IR drop nodes

        ri = torch.randint(0, n_top, (self.n_pairs,), device=device)
        rj = torch.randint(0, n_bot, (self.n_pairs,), device=device)
        hard_i, hard_j = top_idx[ri], bot_idx[rj]

        rand_i = torch.randint(0, N, (self.n_pairs,), device=device)
        rand_j = torch.randint(0, N, (self.n_pairs,), device=device)

        total_loss = torch.tensor(0.0, device=device)
        n_terms = 0
        for idx_i, idx_j in [(hard_i, hard_j), (rand_i, rand_j)]:
            diff_score = scores[idx_i] - scores[idx_j]
            diff_label = labels[idx_i] - labels[idx_j]
            mask = diff_label.abs() > 1e-9
            if mask.sum() == 0:
                continue
            target = (diff_label[mask] > 0).float()
            total_loss = total_loss + torch.nn.functional.binary_cross_entropy_with_logits(
                diff_score[mask], target
            )
            n_terms += 1
        return total_loss / max(n_terms, 1)


class IRDropGNN(nn.Module):
    def __init__(self, node_dim=3, edge_dim=6, hidden=64, n_layers=8, dropout=0.1):
        super().__init__()
        self.node_enc = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
        )
        self.convs = nn.ModuleList([
            PDNConv(hidden, edge_dim, hidden, dropout) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, x, edge_index, edge_attr, demand_mask):
        h = self.node_enc(x)
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
        h = self.dropout(h)
        return self.readout(h[demand_mask]).squeeze(-1)  # (N_demand,)