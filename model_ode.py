"""
Physics-based quasi-static IR drop model.

Learns R_ij from edge geometry via a small MLP.
Forward pass: build Laplacian → solve L_free @ V = rhs → IR drop.
Fully differentiable through torch.linalg.solve.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResistanceMLP(nn.Module):
    """Predict resistance R_ij from 6-dim edge features."""
    def __init__(self, in_dim: int = 6, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        with torch.no_grad():
            self.net[-1].bias.fill_(-2.944)  

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(edge_attr)).squeeze(-1)   


class PDNQuasiStatic(nn.Module):
    """
    Quasi-static IR drop solver via learned resistance.

    For each time step t:
        L_free @ V_free(t) = -I_demand(t) - L_fs @ V_supply
        IR_drop(t)         = V_supply - V_free(t)

    Solves all T time steps in one batched torch.linalg.solve call.
    """

    def __init__(self, hidden: int = 64, V_supply: float = 0.81):
        super().__init__()
        self.R_mlp    = ResistanceMLP(in_dim=6, hidden=hidden)
        self.V_supply = V_supply

    # ------------------------------------------------------------------
    def _laplacian(self, G: torch.Tensor,
                   src: torch.Tensor, dst: torch.Tensor,
                   N_w: int) -> torch.Tensor:
        """Build N_w × N_w graph Laplacian from conductances G (E,)."""
        # off-diagonal: L[i,j] = L[j,i] = -G_ij
        # diagonal:     L[i,i] += G_ij  for every edge touching i
        i_all = torch.cat([src, dst, src, dst])
        j_all = torch.cat([dst, src, src, dst])
        v_all = torch.cat([-G, -G,  G,  G])
        L = torch.zeros(N_w, N_w, dtype=G.dtype, device=G.device)
        L.index_put_((i_all, j_all), v_all, accumulate=True)
        return L

    # ------------------------------------------------------------------
    def forward(self,
                edge_attr:   torch.Tensor,   # (E_wire, 6)
                src:         torch.Tensor,   # (E_wire,) wire-node indices
                dst:         torch.Tensor,   # (E_wire,)
                N_w:         int,
                free_ids:    torch.Tensor,   # (N_free,) LongTensor
                supply_ids:  torch.Tensor,   # (N_sup,)  LongTensor
                I_inj:       torch.Tensor,   # (N_w, T)  current injection (A)
                ) -> torch.Tensor:           # (N_free, T) IR drop (V)

        # 1. predict resistance → conductance
        R = self.R_mlp(edge_attr)            # (E,)
        G = 1.0 / (R + 1e-8)

        # 2. build Laplacian and extract submatrices
        L      = self._laplacian(G, src, dst, N_w)
        L_free = L[free_ids][:, free_ids]
        L_fs   = L[free_ids][:, supply_ids]

        # regularisation: virtual ground conductance handles floating nodes
        eps = 1e-4
        L_free = L_free + torch.eye(
            len(free_ids), dtype=G.dtype, device=G.device) * eps

        # 3. RHS:  -I_demand(t) - L_fs @ V_supply   shape (N_free, T)
        vs = torch.full((len(supply_ids),), self.V_supply,
                        dtype=G.dtype, device=G.device)
        rhs = -I_inj[free_ids] - (L_fs @ vs).unsqueeze(1)

        # 4. solve for all T simultaneously  (N_free, T)
        V_free, _, _, _ = torch.linalg.lstsq(L_free, rhs)
        IR_drop  = self.V_supply - V_free

        return IR_drop   # (N_free, T)
