"""
dynamic_sim.py

Full-chip transient IR-drop simulation — lumped-RC telegrapher model.

  C · dV/dt + G · V = I(t)   (free nodes)
  V_supply = VDD              (pinned)

Backward (implicit) Euler:
  (C/dt + G_FF) · V[n+1] = C/dt · V[n] + I_F[n+1] - G_FS · V_supply

System matrix A = C/dt + G_FF is constant → factorised once with splu,
then one triangular solve per time step.
"""

import os, sys, re, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from telegrapher_ir_drop import (
    parse_def, split_segments_at_intersections, NodeMap,
    build_conductance_matrix, find_supply_nodes, segment_resistance,
    LAYER_PARAMS,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEF_FILE = "826-RISCY-a-2-c2-u0.8-m4-p8-f0.def"
VDD      = 0.81     # V
I_TOTAL  = 0.5      # A  (average total chip current)

# ── Time-domain parameters (realistic 28 nm sign-off conditions) ─────────────
# RISCY target freq: 500 MHz  →  T_clock = 2 ns
# dt = 10 ps : resolves 28 nm switching transient (rise/fall ~20-30 ps)
# T_sim = 10 ns = 5 full clock cycles : enough to see settling
F_CLOCK  = 500e6            # Hz
T_CLOCK  = 1.0 / F_CLOCK   # 2 ns
DT       = 10e-12           # 10 ps  (industry standard for dynamic IR at 28 nm)
T_SIM    = 5 * T_CLOCK      # 10 ns
N_T      = int(round(T_SIM / DT))  # 1000 steps

# ── Current waveform parameters (clock-edge switching model) ─────────────────
# ALPHA = switching activity factor: fraction of FFs that switch per edge.
# Industry typical: 0.1 – 0.3.  0.2 is the standard IR-drop sign-off value.
#
#   I(t) = I_BASE  +  spike(t mod T_CLOCK)
#   spike: triangular, rise/fall = T_RISE = T_CLOCK/20 = 100 ps
#   I_BASE  = (1 - ALPHA) * I_TOTAL   (non-switching cells: leakage + comb.)
#   I_PEAK derived from charge conservation so <I(t)> = I_TOTAL over one cycle:
#     <spike> = (I_PEAK + I_BASE)/2 * 2*T_RISE / T_CLOCK
#     I_TOTAL = I_BASE + <spike>
#     → I_PEAK = I_BASE + (I_TOTAL - I_BASE) * T_CLOCK / T_RISE
I_LEAK = 0.2 * I_TOTAL
I_SWITCH_AVG = I_TOTAL - I_LEAK
ALPHA    = 0.20                      # switching activity factor
T_RISE   = T_CLOCK / 20             # 100 ps rise / fall time
I_BASE = I_LEAK + (1 - ALPHA) * I_SWITCH_AVG  # 0.40 A  — non-switching background
# charge-conserving peak (average over cycle = I_TOTAL)
I_SPIKE_ENERGY = ALPHA * I_SWITCH_AVG * T_CLOCK
I_PEAK = I_BASE + I_SPIKE_ENERGY / T_RISE  # ≈ 1.4 A

# ── Wire capacitance model (28 nm, total line cap fF/μm) ─────────────────────
_CAP_UM = {
    "M1": 0.25e-15, "M2": 0.22e-15, "M3": 0.20e-15,
    "M4": 0.18e-15, "M5": 0.16e-15, "M6": 0.14e-15,
    "M7": 0.12e-15, "M8": 0.10e-15, "M9": 0.08e-15,
}
_CAP_DEFAULT = 0.15e-15   # F/μm


def _seg_cap(seg: dict, dbu_per_um: float) -> float:
    """Total capacitance of one sub-segment [F]."""
    L_um = (abs(seg["x2"] - seg["x1"]) + abs(seg["y2"] - seg["y1"])) / dbu_per_um
    return _CAP_UM.get(seg["layer"].upper(), _CAP_DEFAULT) * L_um


# ── COMPONENTS parser (same heuristic weights as full_sim) ───────────────────
_CELL_WEIGHT_RULES = [
    (re.compile(r"SRAM|RAM|ROM",          re.I), 800.0),
    (re.compile(r"PLL|OSC|CLK_GEN",       re.I), 500.0),
    (re.compile(r"CKN|CKBUF|CLKBUF|CK_", re.I),  12.0),
    (re.compile(r"BUFF?_x16|INV_x16|x16", re.I),  16.0),
    (re.compile(r"BUFF?_x8|INV_x8|x8",   re.I),   8.0),
    (re.compile(r"BUFF?_x4|INV_x4|x4",   re.I),   4.0),
    (re.compile(r"BUFF?_x2|INV_x2|x2",   re.I),   2.0),
    (re.compile(r"FF|DFF|LATCH|REG",      re.I),   3.0),
]


def _cell_weight(ctype: str) -> float:
    for pat, w in _CELL_WEIGHT_RULES:
        if pat.search(ctype):
            return w
    return 1.0


def parse_components(path: str, dbu: float) -> list:
    print("[comp] Parsing COMPONENTS ...", flush=True)
    t0 = time.perf_counter()
    comps, in_comp, pending = [], False, None
    placed_re = re.compile(r"[+]\s+(?:PLACED|FIXED)\s+\(\s*([-\d]+)\s+([-\d]+)\s*\)")
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("COMPONENTS "):
                in_comp = True; continue
            if line.startswith("END COMPONENTS"):
                break
            if not in_comp:
                continue
            if line.startswith("- "):
                parts = line.split()
                if len(parts) >= 3:
                    pending = (parts[1], parts[2])
            m = placed_re.search(line)
            if m and pending:
                inst, ctype = pending
                comps.append({
                    "x_um": int(m.group(1)) / dbu,
                    "y_um": int(m.group(2)) / dbu,
                    "weight": _cell_weight(ctype),
                })
                pending = None
    print(f"[comp] {len(comps):,} components  ({(time.perf_counter()-t0)*1e3:.0f} ms)")
    return comps


def map_cells_to_nodes(comps, nm: NodeMap, dbu: float) -> np.ndarray:
    """Static (average) current injection vector [A], negative = sink."""
    coords = nm.coords()
    xy_um  = np.array([[x / dbu, y / dbu] for x, y in coords])
    tree   = cKDTree(xy_um)
    cell_xy = np.array([[c["x_um"], c["y_um"]] for c in comps])
    weights  = np.array([c["weight"]            for c in comps])
    _, nn    = tree.query(cell_xy, workers=-1)
    i_raw    = np.zeros(len(nm))
    for idx, w in zip(nn, weights):
        i_raw[idx] += w
    total = i_raw.sum()
    return -(i_raw / total) * I_TOTAL if total > 0 else i_raw


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    wall_t0 = time.perf_counter()
    print("=" * 64)
    print("  FULL-CHIP TRANSIENT IR-DROP  —  28 nm / 810 mV")
    print(f"  f_clk={F_CLOCK/1e6:.0f} MHz  T_sim={T_SIM*1e9:.0f} ns  "
          f"dt={DT*1e12:.0f} ps  N_steps={N_T}")
    print("=" * 64)

    # ── PDN topology ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    meta, nets = parse_def(DEF_FILE)
    dbu  = float(meta["dbu_per_micron"])
    segs = nets["VDD"]
    print(f"[pdn]  VDD segments : {len(segs)}")

    sub = split_segments_at_intersections(segs)
    print(f"[pdn]  Sub-segments : {len(sub)}  ({(time.perf_counter()-t0)*1e3:.0f} ms)")

    nm = NodeMap()
    for s in sub:
        nm[(s["x1"], s["y1"])]; nm[(s["x2"], s["y2"])]
    N = len(nm)

    # ── Conductance matrix (G) ────────────────────────────────────────────────
    t0 = time.perf_counter()
    rows, cols, vals = build_conductance_matrix(sub, nm, dbu)
    G_coo = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    print(f"[pdn]  G matrix : {N} × {N}  ({(time.perf_counter()-t0)*1e3:.0f} ms)")

    # ── Nodal capacitance vector (diagonal C) ─────────────────────────────────
    t0 = time.perf_counter()
    C_vec = np.zeros(N)
    for s in sub:
        c_half = 0.5 * _seg_cap(s, dbu)
        C_vec[nm[(s["x1"], s["y1"])]] += c_half
        C_vec[nm[(s["x2"], s["y2"])]] += c_half
    C_vec = np.maximum(C_vec, 1e-18)   # floor to avoid zero caps
    print(f"[pdn]  C_total : {C_vec.sum()*1e12:.2f} pF  "
          f"({(time.perf_counter()-t0)*1e3:.0f} ms)")

    # ── Supply & free node partitioning ──────────────────────────────────────
    supply_ids  = np.array(find_supply_nodes(segs, nm, meta["die"]), dtype=int)
    supply_set  = set(supply_ids.tolist())
    free_ids    = np.array([i for i in range(N) if i not in supply_set], dtype=int)
    N_free      = len(free_ids)
    N_sup       = len(supply_ids)
    print(f"[pdn]  nodes: {N} total  |  {N_sup} supply  |  {N_free} free")

    # ── Sub-matrices ─────────────────────────────────────────────────────────
    G_FF = G_coo[np.ix_(free_ids, free_ids)]
    G_FS = G_coo[np.ix_(free_ids, supply_ids)]
    C_F  = C_vec[free_ids]
    V_sup = np.full(N_sup, VDD)

    # ── Current injection (static distribution, time-scaled below) ───────────
    comps = parse_components(DEF_FILE, dbu)
    I_dc  = map_cells_to_nodes(comps, nm, dbu)   # (N,)  negative = sink
    I_F_dc = I_dc[free_ids]                       # (N_free,)

    # ── System matrix A = C/dt + G_FF  (factorised once) ─────────────────────
    print("[dyn]  Building & factorising system matrix ...", flush=True)
    t0 = time.perf_counter()
    C_diag = sp.diags(C_F / DT, format="csr")
    A      = C_diag + G_FF
    # small shunt for numerical stability on isolated nodes
    A      = A + sp.diags(np.full(N_free, 1e-9), format="csr")
    lu     = spla.splu(A.tocsc())
    rhs_supply = -(G_FS @ V_sup)                  # constant part from supply
    t_factor = time.perf_counter() - t0
    print(f"[dyn]  splu factorisation : {t_factor*1e3:.1f} ms")

    # ── Pre-compute realistic clock-edge current waveform ────────────────────
    # Triangular spike at every rising clock edge:
    #   - background  I_BASE  between edges
    #   - rises linearly from I_BASE to I_PEAK over T_RISE
    #   - falls back to I_BASE over T_RISE
    # Spatial distribution I_F_dc gives per-node weighting (sums to I_TOTAL).
    # Scale factor s(t) : I_node(t) = I_F_dc * s(t)
    t_vec   = np.arange(N_T, dtype=np.float64) * DT
    t_phase = np.mod(t_vec, T_CLOCK)            # time within each clock cycle

    # Normalise I_F_dc so its sum = 1 (shape vector)
    I_shape = I_F_dc / np.sum(np.abs(I_F_dc))

    # Scale s(t): triangle spike 0..T_RISE→ peak, T_RISE..2*T_RISE→ decay
    s = np.where(
        t_phase < T_RISE,
        I_BASE + (I_PEAK - I_BASE) * (t_phase / T_RISE),
        np.where(
            t_phase < 2 * T_RISE,
            I_PEAK - (I_PEAK - I_BASE) * ((t_phase - T_RISE) / T_RISE),
            I_BASE,
        ),
    )
    # s(t) is a scalar multiplier; I_node(t) = I_shape * s(t) [current sink → negative]
    # I_F_dc already negative (sink), so keep sign
    sign = np.sign(I_F_dc)

    # ── Time integration — backward Euler ─────────────────────────────────────
    print(f"[dyn]  Time integration  ({N_T} steps × {DT*1e12:.0f} ps) ...",
          flush=True)
    V_F  = np.full(N_free, VDD)    # IC: all nodes start at VDD
    t0   = time.perf_counter()
    V_all = np.empty((N_T, N_free), dtype=np.float32)

    for k in range(N_T):
        I_F_t = I_shape * s[k]   # (N_free,) current at step k
        rhs   = (C_F / DT) * V_F + I_F_t + rhs_supply
        V_F   = lu.solve(rhs)
        V_all[k] = V_F

    t_integrate = time.perf_counter() - t0
    print(f"[dyn]  Integration done  : {t_integrate:.3f} s  "
          f"({t_integrate/N_T*1e3:.2f} ms/step)")

    # ── Results ───────────────────────────────────────────────────────────────
    IR_all = VDD - V_all          # (N_T, N_free)  IR drop at each node & time
    IR_max_over_time = IR_all.max(axis=0)   # worst drop at each node across all t
    IR_max_over_node = IR_all.max(axis=1)   # worst node drop at each time step

    worst_node_ir = IR_max_over_time.max()
    worst_t_idx   = int(IR_max_over_node.argmax())

    print()
    print("=" * 64)
    print("  TRANSIENT IR DROP REPORT")
    print("=" * 64)
    print(f"  Free nodes         : {N_free}")
    print(f"  Clock / dt / T_sim : {F_CLOCK/1e6:.0f} MHz / {DT*1e12:.0f} ps / {T_SIM*1e9:.0f} ns")
    print(f"  Time steps         : {N_T}")
    print(f"  Max IR drop (ever) : {worst_node_ir*1e3:.2f} mV")
    print(f"  Mean IR drop (t=worst): {IR_all[worst_t_idx].mean()*1e3:.2f} mV")
    print(f"  Worst time step    : {worst_t_idx * DT * 1e9:.2f} ns")
    print(f"  Min voltage (ever) : {(VDD - worst_node_ir)*1e3:.2f} mV  "
          f"({(VDD-worst_node_ir)/VDD*100:.2f}% of VDD)")
    print("=" * 64)

    total_s = time.perf_counter() - wall_t0
    print(f"\n[time] splu factorisation : {t_factor*1e3:.1f} ms")
    print(f"[time] time integration   : {t_integrate:.3f} s  ({N_T} steps)")
    print(f"[time] total wall time    : {total_s:.2f} s")


if __name__ == "__main__":
    main()
