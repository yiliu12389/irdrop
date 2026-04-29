"""IR Drop GNN Training Script."""

import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from scipy.stats import spearmanr

from config import (
    BASE_DIR, LEF_PATH, LABEL_DIR, POWER_DIR, CKPT_PATH,
    HIDDEN, N_LAYERS, NODE_DIM, DROPOUT, LR, EPOCHS,
    SPLIT_SEED, DEVICE, HOT_THRESH, WEIGHT_DECAY,
)
from dataset import load_sample, make_splits, design_from_name
from model import IRDropGNN, PairwiseRankingLoss
from parsers import LEFParser


# ─── Per-sample train/eval step ───────────────────────────────────────────────

def _ranking_metrics(scores: torch.Tensor, ir: torch.Tensor, k: int = 100):
    """Spearman ρ (all samples) + Recall@k (only when hotspots exist)."""
    s = scores.cpu().numpy()
    t = ir.cpu().numpy()
    rho = float(spearmanr(s, t).correlation)

    n_hot = int((t > HOT_THRESH).sum())
    if n_hot == 0:
        return rho, float("nan")      # no hotspots → skip recall@k
    k_eff       = min(k, n_hot)
    top_pred    = set(np.argsort(s)[-k_eff:])
    top_true    = set(np.argsort(t)[-k_eff:])
    return rho, len(top_pred & top_true) / k_eff


def run_one(model, data, label, loss_fn, optimizer, device, train=True):
    x           = data.x.to(device)
    edge_index  = data.edge_index.to(device)
    edge_attr   = data.edge_attr.to(device)
    demand_mask = data.demand_mask.to(device)
    lbl         = label.to(device)

    model.train() if train else model.eval()
    if train:
        optimizer.zero_grad()

    with torch.set_grad_enabled(train):
        scores = model(x, edge_index, edge_attr, demand_mask)
        loss   = loss_fn(scores, lbl)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    with torch.no_grad():
        rho, recall_at_k = _ranking_metrics(scores, lbl)

    return loss.item(), rho, recall_at_k


# ─── DDP helpers ──────────────────────────────────────────────────────────────

def _sync_scalar(val: float, count: int, device) -> float:
    t = torch.tensor([val * count, float(count)], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t[0] / t[1]).item() if t[1].item() > 0 else float("nan")


def _avg_model_params(model: nn.Module) -> None:
    world_size = dist.get_world_size()
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= world_size


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    use_dist = dist.is_available() and "RANK" in os.environ
    if use_dist:
        dist.init_process_group(backend="nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        device     = f"cuda:{rank}"
        torch.cuda.set_device(rank)
    else:
        rank       = 0
        world_size = 1
        device     = DEVICE

    is_master = (rank == 0)

    if is_master:
        print("Parsing LEF …")
    lef_data = LEFParser(str(LEF_PATH)).parse()
    if is_master:
        print(f"  layers={len(lef_data[0])}, macros={len(lef_data[1])}")

    all_names = sorted(os.listdir(POWER_DIR))
    all_names = [n for n in all_names if (LABEL_DIR / n).exists()]
    if use_dist:
        if is_master:
            make_splits(all_names)
        dist.barrier()
        splits  = json.loads((BASE_DIR / "splits.json").read_text())
        trn_set = splits["train"]
        val_set = splits["val"]
        tst_set = splits["test"]
    else:
        trn_set, val_set, tst_set = make_splits(all_names)
        if is_master:
            print(f"Splits: train={len(trn_set)}  val={len(val_set)}  test={len(tst_set)}")

    model     = IRDropGNN(node_dim=NODE_DIM, edge_dim=6, hidden=HIDDEN,
                          n_layers=N_LAYERS, dropout=DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn   = PairwiseRankingLoss(n_pairs=1024, top_frac=0.10, bot_frac=0.50)

    if is_master:
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Loss: PairwiseRanking (top10% vs bot50%, all samples)")

    best_val_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        ep_start = time.time()
        rng_ep   = random.Random(SPLIT_SEED + epoch)
        trn_shuffled = trn_set[:]
        rng_ep.shuffle(trn_shuffled)
        trn_local = trn_shuffled[rank::world_size]

        # ── Train ──
        trn_losses, trn_rhos, trn_recs = [], [], []
        for idx, name in enumerate(trn_local):
            try:
                data, label = load_sample(name, lef_data)
                loss, rho, rec = run_one(
                    model, data, label, loss_fn, optimizer, device, train=True)
                trn_losses.append(loss)
                if not np.isnan(rho):   trn_rhos.append(rho)
                if not np.isnan(rec):   trn_recs.append(rec)
            except Exception as e:
                if is_master:
                    print(f"  [skip train] {name}: {e}")
                continue
            if is_master and (idx + 1) % 100 == 0:
                print(f"  epoch {epoch}  {idx+1}/{len(trn_local)}"
                      f"  loss={np.mean(trn_losses):.4f}"
                      f"  spearman={np.mean(trn_rhos):.3f}"
                      f"  recall@k={np.mean(trn_recs)*100:.1f}%")

        if use_dist:
            _avg_model_params(model)

        # ── Val ──
        val_losses, val_rhos, val_recs = [], [], []
        for name in val_set[rank::world_size]:
            try:
                data, label = load_sample(name, lef_data)
                loss, rho, rec = run_one(
                    model, data, label, loss_fn, None, device, train=False)
                val_losses.append(loss)
                if not np.isnan(rho): val_rhos.append(rho)
                if not np.isnan(rec): val_recs.append(rec)
            except Exception as e:
                if is_master:
                    print(f"  [skip val] {name}: {e}")

        scheduler.step()

        def _agg(lst): return float(np.mean(lst)) if lst else float("nan")

        if use_dist:
            trn_loss = _sync_scalar(_agg(trn_losses), len(trn_losses), device)
            trn_rho  = _sync_scalar(_agg(trn_rhos),   len(trn_rhos),   device)
            trn_rec  = _sync_scalar(_agg(trn_recs),   len(trn_recs),   device)
            val_loss = _sync_scalar(_agg(val_losses), len(val_losses), device)
            val_rho  = _sync_scalar(_agg(val_rhos),   len(val_rhos),   device)
            val_rec  = _sync_scalar(_agg(val_recs),   len(val_recs),   device)
        else:
            trn_loss = _agg(trn_losses)
            trn_rho  = _agg(trn_rhos)
            trn_rec  = _agg(trn_recs)
            val_loss = _agg(val_losses)
            val_rho  = _agg(val_rhos)
            val_rec  = _agg(val_recs)

        if is_master:
            elapsed = time.time() - ep_start
            print(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"trn loss={trn_loss:.4f}  ρ={trn_rho:.3f}  rec@k={trn_rec*100:.1f}% | "
                f"val loss={val_loss:.4f}  ρ={val_rho:.3f}  rec@k={val_rec*100:.1f}% | "
                f"{elapsed:.0f}s"
            )
            score = val_rho if not np.isnan(val_rho) else 0.0
            if score > best_val_f1:
                best_val_f1 = score
                torch.save({
                    "epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_spearman": val_rho, "val_recall_at_k": val_rec,
                }, CKPT_PATH)
                print(f"  ✓ checkpoint  val_ρ={val_rho:.3f}  recall@k={val_rec*100:.1f}%")

    if use_dist:
        dist.barrier()

    # ── Test ──
    if is_master and tst_set:
        print(f"\nBest val_F1 = {best_val_f1*100:.2f}%  — loading checkpoint for test …")
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model"])

        tst_by_design: dict = {}
        for name in tst_set:
            tst_by_design.setdefault(design_from_name(name), []).append(name)

        print(f"\n{'Design':<20} {'N':>5}  {'Spearman ρ':>11}  {'Recall@k (%)':>13}")
        print(f"{'-'*20}  {'-'*5}  {'-'*11}  {'-'*13}")
        all_rhos, all_recs = [], []
        for design in sorted(tst_by_design):
            d_rhos, d_recs = [], []
            for name in tst_by_design[design]:
                try:
                    data, label = load_sample(name, lef_data)
                    _, rho, rec = run_one(
                        model, data, label, loss_fn, None, device, train=False)
                    if not np.isnan(rho): d_rhos.append(rho)
                    if not np.isnan(rec): d_recs.append(rec)
                except Exception as e:
                    print(f"  [skip test] {name}: {e}")
            if d_rhos:
                print(f"{design:<20} {len(d_rhos):>5}"
                      f"  {np.mean(d_rhos):>11.3f}"
                      f"  {np.mean(d_recs)*100:>13.1f}")
                all_rhos.extend(d_rhos); all_recs.extend(d_recs)
        print(f"{'-'*20}  {'-'*5}  {'-'*11}  {'-'*13}")
        print(f"{'TOTAL':<20} {len(all_rhos):>5}"
              f"  {np.mean(all_rhos):>11.3f}"
              f"  {np.mean(all_recs)*100:>13.1f}")

    if use_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
