"""
Training loop for the physics-based quasi-static IR drop model.

One sample = one forward solve (no batching across graphs).
Differentiates through torch.linalg.solve to learn R_ij from data.
"""

import os, sys, time, json, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent))
from config import (LEF_PATH, DEF_DIR, POWER_DIR, LABEL_DIR,
                    POWER_NETS, TARGET_NET, HOT_THRESH, DEVICE)
from graph import PDNGraphBuilder
from parsers import LEFParser
from model_ode import PDNQuasiStatic

# ── hyper-params ───────────────────────────────────────────────────────────
HIDDEN      = 64
V_SUPPLY    = 0.81
EPOCHS      = 30
LR          = 3e-4
SUPPLY_PCTILE = 95  
MAX_SAMPLES = None   


# ── helpers ────────────────────────────────────────────────────────────────

def load_lef():
    return LEFParser(str(LEF_PATH)).parse()


def find_def(name: str) -> Path:
    for sfx in [".def.gz", ".def"]:
        p = DEF_DIR / (name + sfx)
        if p.exists():
            return p
    # also check uncompressed in project root
    p = Path(__file__).parent / (name + ".def")
    if p.exists():
        return p
    raise FileNotFoundError(f"DEF not found: {name}")


def prepare_sample(name: str, lef_data, device):
    """
    Build graph, extract tensors needed for the ODE forward pass.

    Returns dict with:
        edge_attr   (E_wire, 6)   features of wire+via edges
        src, dst    (E_wire,)     wire-node indices (0..N_w-1)
        N_w         int
        free_ids    (N_free,)
        supply_ids  (N_sup,)
        I_inj       (N_w, T)     current injection at each wire node
        demand_wire (N_dem,)     wire-node index for each demand node
        ir_gt       (N_dem,)     ground-truth IR drop in V
    """
    def_path   = find_def(name)
    power_path = str(POWER_DIR / name)
    label_path = str(LABEL_DIR / name)

    builder = PDNGraphBuilder(
        str(LEF_PATH), str(def_path), power_path,
        power_nets=POWER_NETS, target_net=TARGET_NET,
        _shared_lef=lef_data)
    G    = builder.build()
    data = builder.to_pyg(G)

    nodes   = list(G.nodes())
    nidx    = {n: i for i, n in enumerate(nodes)}
    is_dem  = np.array([G.nodes[n].get("node_type", 0) == 1.0 for n in nodes])
    wids    = np.where(~is_dem)[0]
    N_w     = len(wids)
    wset    = set(wids.tolist())

    # ── wire+via conductance edges ──────────────────────────────────────
    e_src, e_dst, e_attr = [], [], []
    for u, v, edata in G.edges(data=True):
        iu, iv = nidx[u], nidx[v]
        if iu not in wset or iv not in wset:
            continue
        wu = int(np.searchsorted(wids, iu))
        wv = int(np.searchsorted(wids, iv))
        feat = [
            edata.get("length",    0.0),
            edata.get("width",     0.0),
            edata.get("layer_idx", 0.0),
            edata.get("cut_area",  0.0),
            edata.get("num_cuts",  0.0),
            0.0 if edata.get("edge_type") == "wire" else 1.0,
        ]
        e_src.append(wu); e_dst.append(wv); e_attr.append(feat)

    if not e_src:
        return None

    src       = torch.tensor(e_src,  dtype=torch.long)
    dst       = torch.tensor(e_dst,  dtype=torch.long)
    edge_attr = torch.tensor(e_attr, dtype=torch.float32)

    # ── supply nodes: M7 + M8 wire nodes (top-layer power grid) ─────────
    # Normalized layer_idx: M7 ≈ 0.70, M8 ≈ 0.80  (depends on LEF layers)
    # Use threshold 0.65 to reliably capture both.
    wire_li = torch.tensor(
        [G.nodes[nodes[wids[i]]].get("layer_idx", 0.0) for i in range(N_w)],
        dtype=torch.float32)
    supply_mask = wire_li > 0.65
    if supply_mask.sum() < 10:          # fallback: top-5% by degree
        deg = torch.zeros(N_w)
        deg.scatter_add_(0, src, torch.ones(len(src)))
        deg.scatter_add_(0, dst, torch.ones(len(dst)))
        k = max(10, int(0.05 * N_w))
        supply_mask = torch.zeros(N_w, dtype=torch.bool)
        supply_mask[deg.topk(k).indices] = True
    supply_ids = torch.where(supply_mask)[0]
    free_mask  = ~supply_mask
    free_ids   = torch.where(free_mask)[0]

    # ── current injection: aggregate demand power onto wire nodes ───────
    T_steps = 20
    I_inj   = torch.zeros(N_w, T_steps)
    demand_wire_list = []   # wire-node idx per demand node
    for u, v, edata in G.edges(data=True):
        iu, iv = nidx[u], nidx[v]
        if edata.get("edge_type") != "inject":
            continue
        d_idx, w_idx = (iu, iv) if is_dem[iu] else (iv, iu)
        if w_idx not in wset:
            continue
        wi = int(np.searchsorted(wids, w_idx))
        p  = G.nodes[nodes[d_idx]].get("power")
        if p is not None:
            I_inj[wi] += torch.tensor(p, dtype=torch.float32) / V_SUPPLY
        demand_wire_list.append((d_idx, wi))

    # ── ground-truth IR drop (V) per demand node ────────────────────────
    ir_raw = np.load(label_path)
    ir_2d  = ir_raw.max(axis=0) if ir_raw.ndim == 3 else ir_raw
    tile_map = builder.demand_tile_map

    ir_gt_vals, demand_wire_ids = [], []
    for d_idx, wi in demand_wire_list:
        cell_node = nodes[d_idx]
        if cell_node not in tile_map:
            continue
        tiles  = tile_map[cell_node]
        max_ir = max(float(ir_2d[r, c]) for r, c in tiles)
        ir_gt_vals.append(max_ir / 1000.0)   # mV → V
        demand_wire_ids.append(wi)

    if not ir_gt_vals:
        return None

    ir_gt       = torch.tensor(ir_gt_vals,   dtype=torch.float32)
    demand_wire = torch.tensor(demand_wire_ids, dtype=torch.long)

    return dict(
        edge_attr=edge_attr.to(device),
        src=src.to(device),
        dst=dst.to(device),
        N_w=N_w,
        free_ids=free_ids.to(device),
        supply_ids=supply_ids.to(device),
        I_inj=I_inj.to(device),
        demand_wire=demand_wire,   # stay CPU for indexing
        ir_gt=ir_gt.to(device),
        name=name,
    )


def predict_demand_ir(ir_free: torch.Tensor,
                      free_ids: torch.Tensor,
                      demand_wire: torch.Tensor,
                      N_w: int) -> torch.Tensor:
    """
    Map per-free-node IR drop (N_free, T) to per-demand-node IR drop (N_dem,).
    Uses max-over-T as the prediction target (matches GT construction).
    """
    # build N_w -> free_idx lookup
    free_ids_cpu = free_ids.cpu()
    fw_map = torch.full((N_w,), -1, dtype=torch.long)
    fw_map[free_ids_cpu] = torch.arange(len(free_ids_cpu))

    fi = fw_map[demand_wire]           # free-node index for each demand node
    valid = fi >= 0
    pred = torch.zeros(len(demand_wire), device=ir_free.device)
    if valid.any():
        pred[valid] = ir_free[fi[valid].to(ir_free.device)].max(dim=1).values
    return pred


# ── training ───────────────────────────────────────────────────────────────

def main():
    device = torch.device(DEVICE)
    print(f"device: {device}")

    # split file
    split_path = Path(__file__).parent / "splits.json"
    if split_path.exists():
        splits    = json.load(open(split_path))
        train_ids = splits["train"]
        val_ids   = splits["val"]
    else:
        all_names = [f.stem for f in LABEL_DIR.iterdir()]
        random.seed(42); random.shuffle(all_names)
        k = int(len(all_names) * 0.85)
        train_ids, val_ids = all_names[:k], all_names[k:]

    if MAX_SAMPLES:
        train_ids = train_ids[:MAX_SAMPLES]
        val_ids   = val_ids[:5]

    print(f"train={len(train_ids)}  val={len(val_ids)}")

    lef_data = load_lef()
    model    = PDNQuasiStatic(hidden=HIDDEN, V_supply=V_SUPPLY).to(device)
    opt      = Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched    = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.01)
    loss_fn  = nn.HuberLoss(delta=0.01)

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ── train ──────────────────────────────────────────────────────
        model.train()
        random.shuffle(train_ids)
        train_loss, n_ok = 0.0, 0

        for name in train_ids:
            try:
                s = prepare_sample(name, lef_data, device)
            except Exception as e:
                continue
            if s is None:
                continue

            opt.zero_grad()
            ir_free = model(s["edge_attr"], s["src"], s["dst"],
                            s["N_w"], s["free_ids"], s["supply_ids"],
                            s["I_inj"])                               # (N_free, T)

            pred = predict_demand_ir(ir_free, s["free_ids"],
                                     s["demand_wire"], s["N_w"])
            loss = loss_fn(pred, s["ir_gt"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss += loss.item(); n_ok += 1

        sched.step()
        avg_train = train_loss / max(n_ok, 1)

        # ── val ────────────────────────────────────────────────────────
        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for name in val_ids:
                try:
                    s = prepare_sample(name, lef_data, device)
                except Exception:
                    continue
                if s is None:
                    continue
                ir_free = model(s["edge_attr"], s["src"], s["dst"],
                                s["N_w"], s["free_ids"], s["supply_ids"],
                                s["I_inj"])
                pred    = predict_demand_ir(ir_free, s["free_ids"],
                                            s["demand_wire"], s["N_w"])
                val_loss += loss_fn(pred, s["ir_gt"]).item(); n_val += 1

        avg_val = val_loss / max(n_val, 1)
        print(f"epoch {epoch:3d}  train={avg_train:.5f}  val={avg_val:.5f}  "
              f"lr={sched.get_last_lr()[0]:.2e}  ok={n_ok}/{len(train_ids)}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), "irdrop_ode.pt")
            print(f"           -> saved (val={best_val:.5f})")

    print(f"Done. Best val loss: {best_val:.5f}")


if __name__ == "__main__":
    main()
