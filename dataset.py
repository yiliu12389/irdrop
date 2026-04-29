"""Sample loading and dataset splitting."""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from config import (
    BASE_DIR, DEF_DIR, POWER_DIR, LABEL_DIR, LEF_PATH,
    POWER_NETS, TARGET_NET, HOT_THRESH,
    TEST_DESIGNS, VAL_RATIO, SPLIT_SEED, _DESIGN_RE,
)
from graph import PDNGraphBuilder


def design_from_name(name: str) -> str:
    m = _DESIGN_RE.match(name)
    return m.group(1) if m else "unknown"


def load_sample(name: str, lef_data):
    """Build PDN graph and return (data, label).

    data.x          : (N, 3)  [layer_idx, node_type, max_power]
    data.demand_mask: (N,) bool
    label           : (N_demand,) float  1 = hotspot (max IR drop > HOT_THRESH)
    """
    power_path = POWER_DIR / name
    label_path = LABEL_DIR / name

    for suffix in [".def.gz", ".def"]:
        def_path = DEF_DIR / (name + suffix)
        if def_path.exists():
            break
    else:
        raise FileNotFoundError(f"DEF not found for {name}")

    builder = PDNGraphBuilder(
        lef_path=LEF_PATH, def_path=def_path,
        power_path=str(power_path),
        power_nets=POWER_NETS, target_net=TARGET_NET,
        _shared_lef=lef_data,
    )
    G    = builder.build()
    data = builder.to_pyg(G)

    # Label: max IR drop (µV → V) per demand node, used as ranking target.
    ir_raw = np.load(str(label_path))                        # (H, W) or (T, H, W)
    ir_2d  = ir_raw.max(axis=0) if ir_raw.ndim == 3 else ir_raw  # (H, W)

    tile_map = builder.demand_tile_map   # cell_node → [(row,col), ...]
    labels = []
    for node in builder.demand_node_order:
        tiles  = tile_map[node]
        max_ir = max(ir_2d[r, c] for r, c in tiles)
        labels.append(float(max_ir) / 1000.0)               # µV → V
    label = torch.tensor(labels, dtype=torch.float32)

    return data, label


def make_splits(all_names: List[str]) -> Tuple[List[str], List[str], List[str]]:
    split_file = BASE_DIR / "splits.json"

    if split_file.exists():
        splits = json.loads(split_file.read_text())
        print(f"Loaded existing splits from {split_file}")
        return splits["train"], splits["val"], splits["test"]

    rng = random.Random(SPLIT_SEED)

    by_design: Dict[str, List[str]] = {}
    for n in all_names:
        by_design.setdefault(design_from_name(n), []).append(n)

    trn, val, tst = [], [], []
    print("\nCross-design split  (train: RISCY-*, test: zero-riscy-*)")
    print(f"  {'Design':<20} {'Total':>6}  {'Train':>6}  {'Val':>5}  {'Test':>5}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}")
    for design in sorted(by_design):
        names = by_design[design][:]
        rng.shuffle(names)
        n = len(names)
        if design in TEST_DESIGNS:
            tst += names
            print(f"  {design:<20} {n:>6}  {'—':>6}  {'—':>5}  {n:>5}")
        else:
            n_val = max(1, round(n * VAL_RATIO))
            n_trn = n - n_val
            trn += names[:n_trn]
            val += names[n_trn:]
            print(f"  {design:<20} {n:>6}  {n_trn:>6}  {n_val:>5}  {'—':>5}")
    print(f"  {'TOTAL':<20} {len(all_names):>6}  {len(trn):>6}  {len(val):>5}  {len(tst):>5}\n")

    split_file.write_text(json.dumps({"train": trn, "val": val, "test": tst}, indent=2))
    print(f"Splits saved to {split_file}")
    return trn, val, tst
