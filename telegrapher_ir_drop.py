"""
telegrapher_ir_drop.py

Quasi-static telegrapher's equation solver for PDN IR-drop analysis.

  Y · v = i
  Y[i,i]  = sum of conductances incident on node i  (+ shunt leakage)
  Y[i,j]  = -conductance between nodes i and j
  v[i]    = voltage at node i
  i[i]    = net current injected at node i (positive = source)

Wire resistance:  R = Rsh * L_um / W_um
"""

import re
from collections import defaultdict

# ── Layer sheet-resistance table (28 nm process, Ω/□) ─────────────────────
LAYER_PARAMS = {
    "M1":  {"rsh": 0.080, "via_r": 5.0},
    "M2":  {"rsh": 0.073, "via_r": 4.5},
    "M3":  {"rsh": 0.073, "via_r": 4.0},
    "M4":  {"rsh": 0.027, "via_r": 3.0},
    "M5":  {"rsh": 0.027, "via_r": 3.0},
    "M6":  {"rsh": 0.013, "via_r": 2.0},
    "M7":  {"rsh": 0.013, "via_r": 2.0},
    "M8":  {"rsh": 0.008, "via_r": 1.5},
    "M9":  {"rsh": 0.004, "via_r": 1.5},
    "MRDL":{"rsh": 0.004, "via_r": 1.5},
}
_DEFAULT_RSH = 0.050


def _rsh(layer: str) -> float:
    return LAYER_PARAMS.get(layer.upper(), {}).get("rsh", _DEFAULT_RSH)


# ── Segment resistance ─────────────────────────────────────────────────────

def segment_resistance(seg: dict, dbu_per_um: float) -> float:
    """Return resistance in Ω for a single wire segment."""
    length_dbu = abs(seg["x2"] - seg["x1"]) + abs(seg["y2"] - seg["y1"])
    width_dbu  = seg["width"]
    if length_dbu == 0 or width_dbu == 0:
        return 1e-6
    L_um = length_dbu / dbu_per_um
    W_um = width_dbu  / dbu_per_um
    return max(_rsh(seg["layer"]) * L_um / W_um, 1e-9)


# ── DEF parser (SPECIALNETS + DIEAREA) ────────────────────────────────────

_COORD_RE = re.compile(r"\(\s*([\d*-]+)\s+([\d*-]+)\s*\)")
_NEW_RE   = re.compile(r"(?:ROUTED|NEW)\s+(\S+)\s+([\d]+)")


def parse_def(path: str) -> tuple:
    """
    Returns (meta, nets).
      meta = {"dbu_per_micron": float, "die": {x0,y0,x1,y1}}
      nets = {net_name: [seg_dict, ...]}
        seg_dict keys: layer, width, x1, y1, x2, y2
    """
    meta = {"dbu_per_micron": 1000.0,
            "die": {"x0": 0, "y0": 0, "x1": 0, "y1": 0}}
    nets: dict = defaultdict(list)

    in_sn      = False
    cur_net    = None
    last_x     = None
    last_y     = None
    cur_layer  = None
    cur_width  = 0

    dbu_re = re.compile(r"UNITS\s+DISTANCE\s+MICRONS\s+([\d.]+)")
    die_re = re.compile(
        r"DIEAREA\s+\(\s*([\d-]+)\s+([\d-]+)\s*\)\s+\(\s*([\d-]+)\s+([\d-]+)\s*\)")
    net_re = re.compile(r"^-\s+(\S+)")

    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue

            m = dbu_re.search(line)
            if m:
                meta["dbu_per_micron"] = float(m.group(1))
                continue

            m = die_re.search(line)
            if m:
                meta["die"] = {
                    "x0": int(m.group(1)), "y0": int(m.group(2)),
                    "x1": int(m.group(3)), "y1": int(m.group(4)),
                }
                continue

            if line.startswith("SPECIALNETS "):
                in_sn = True; continue
            if line.startswith("END SPECIALNETS"):
                break
            if not in_sn:
                continue

            m = net_re.match(line)
            if m and not line.startswith("+ ") and "NEW" not in line:
                cur_net = m.group(1)
                last_x = last_y = None
                cur_layer = None
                continue

            if cur_net is None:
                continue

            m = _NEW_RE.search(line)
            if m:
                cur_layer = m.group(1).upper()
                cur_width = int(m.group(2))
                last_x = last_y = None

            for xs, ys in _COORD_RE.findall(line):
                xi = last_x if xs == "*" else int(xs)
                yi = last_y if ys == "*" else int(ys)
                if last_x is not None and cur_layer is not None:
                    if xi != last_x or yi != last_y:
                        nets[cur_net].append({
                            "layer": cur_layer, "width": cur_width,
                            "x1": last_x, "y1": last_y,
                            "x2": xi,     "y2": yi,
                        })
                last_x, last_y = xi, yi

    return meta, dict(nets)


# ── NodeMap ────────────────────────────────────────────────────────────────

class NodeMap:
    """Maps (x, y) integer coordinates → unique integer node index."""

    def __init__(self):
        self._map: dict = {}

    def __getitem__(self, xy: tuple) -> int:
        idx = self._map.get(xy)
        if idx is None:
            idx = len(self._map)
            self._map[xy] = idx
        return idx

    def __len__(self) -> int:
        return len(self._map)

    def coords(self) -> list:
        return list(self._map.keys())


# ── Intersection splitting ─────────────────────────────────────────────────

def split_segments_at_intersections(segs: list) -> list:
    """Split H/V wire segments at every crossing point."""
    h_segs, v_segs = [], []
    for s in segs:
        if s["x1"] == s["x2"]:
            v_segs.append({**s, "x": s["x1"],
                           "y_lo": min(s["y1"], s["y2"]),
                           "y_hi": max(s["y1"], s["y2"])})
        elif s["y1"] == s["y2"]:
            h_segs.append({**s, "y": s["y1"],
                           "x_lo": min(s["x1"], s["x2"]),
                           "x_hi": max(s["x1"], s["x2"])})

    v_by_x: dict = defaultdict(list)
    for vs in v_segs:
        v_by_x[vs["x"]].append(vs)

    h_by_y: dict = defaultdict(list)
    for hs in h_segs:
        h_by_y[hs["y"]].append(hs)

    result = []

    for hs in h_segs:
        y, x0, x1 = hs["y"], hs["x_lo"], hs["x_hi"]
        split_xs = {x0, x1}
        for vx, vlist in v_by_x.items():
            if x0 < vx < x1:
                for vs in vlist:
                    if vs["y_lo"] <= y <= vs["y_hi"]:
                        split_xs.add(vx)
        xs = sorted(split_xs)
        for i in range(len(xs) - 1):
            result.append({"layer": hs["layer"], "width": hs["width"],
                           "x1": xs[i], "y1": y, "x2": xs[i+1], "y2": y})

    for vs in v_segs:
        x, y0, y1 = vs["x"], vs["y_lo"], vs["y_hi"]
        split_ys = {y0, y1}
        for hy, hlist in h_by_y.items():
            if y0 < hy < y1:
                for hs in hlist:
                    if hs["x_lo"] <= x <= hs["x_hi"]:
                        split_ys.add(hy)
        ys = sorted(split_ys)
        for i in range(len(ys) - 1):
            result.append({"layer": vs["layer"], "width": vs["width"],
                           "x1": x, "y1": ys[i], "x2": x, "y2": ys[i+1]})

    return result


# ── Conductance matrix ─────────────────────────────────────────────────────

def build_conductance_matrix(sub_segs: list, nm: NodeMap,
                             dbu_per_um: float) -> tuple:
    """Build COO entries for the symmetric nodal conductance matrix."""
    rows, cols, vals = [], [], []
    for seg in sub_segs:
        ni = nm[(seg["x1"], seg["y1"])]
        nj = nm[(seg["x2"], seg["y2"])]
        if ni == nj:
            continue
        g = 1.0 / segment_resistance(seg, dbu_per_um)
        rows += [ni, nj, ni, nj]
        cols += [nj, ni, ni, nj]
        vals += [-g, -g, g,   g]
    return rows, cols, vals


# ── Supply-node detection ──────────────────────────────────────────────────

def find_supply_nodes(segs: list, nm: NodeMap, die: dict) -> list:
    """Return node indices on or near the die boundary (power ring)."""
    margin_x = (die["x1"] - die["x0"]) * 0.05
    margin_y = (die["y1"] - die["y0"]) * 0.05
    x_lo = die["x0"] + margin_x
    x_hi = die["x1"] - margin_x
    y_lo = die["y0"] + margin_y
    y_hi = die["y1"] - margin_y

    ring = set()
    for xy, idx in nm._map.items():
        x, y = xy
        if x <= x_lo or x >= x_hi or y <= y_lo or y >= y_hi:
            ring.add(idx)
    for seg in segs:
        for xy in [(seg["x1"], seg["y1"]), (seg["x2"], seg["y2"])]:
            x, y = xy
            if (x == die["x0"] or x == die["x1"] or
                    y == die["y0"] or y == die["y1"]):
                if xy in nm._map:
                    ring.add(nm._map[xy])
    return sorted(ring)
