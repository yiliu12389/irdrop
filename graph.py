
import re
import math
import bisect
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data

from parsers import LEFParser, DEFParser


class PDNGraphBuilder:
    TILE_SIZE = 2.25
    NODE_FEAT_NAMES = ["layer_idx", "node_type", "power"]
    EDGE_FEAT_NAMES = ["length", "width", "layer_idx", "cut_area", "num_cuts", "edge_type"]
    _EDGE_TYPE_MAP  = {"wire": 0.0, "via": 1.0, "inject": 2.0}

    def __init__(self, lef_path, def_path, power_path, power_nets, target_net,
                 m1_layer="M1", grid_snap=0.001,
                 _shared_lef=None):
        self.target_net = target_net
        self.m1_layer   = m1_layer
        self.grid_snap  = grid_snap

        if _shared_lef is not None:
            self.layers, self.macros = _shared_lef
        else:
            self.layers, self.macros = LEFParser(lef_path).parse()

        routing = [n for n, l in self.layers.items() if l.direction in ("HORIZONTAL", "VERTICAL")]
        self._layer_order = {
            name: i for i, name in enumerate(sorted(
                routing,
                key=lambda s: int(re.search(r'\d+', s).group()) if re.search(r'\d+', s) else 0
            ))
        }
        self._tile_layer_idx = len(self._layer_order)

        def_parser = DEFParser(def_path, power_nets)
        self.segments, self.vias = def_parser.parse()
        self.die_area = def_parser.die_area
        self.power = np.load(power_path) if power_path else None

    def build(self) -> nx.Graph:
        G = nx.Graph()
        self._add_segment_edges(G)
        self._add_via_edges(G)
        if self.power is not None:
            self._add_demand_node(G, self.power)
        self._prune_seg_endpoints(G)
        self._collapse_via_stacks(G)
        self._collapse_wire_chains(G)
        self._compute_node_features(G)
        return G

    def to_pyg(self, G: nx.Graph) -> Data:
        nodes    = list(G.nodes())
        node_idx = {n: i for i, n in enumerate(nodes)}
        N        = len(nodes)

        x   = torch.zeros(N, len(self.NODE_FEAT_NAMES), dtype=torch.float32)
        pos = torch.zeros(N, 2, dtype=torch.float32)

        for i, node in enumerate(nodes):
            attrs = G.nodes[node]
            x[i] = torch.tensor([
                attrs.get("layer_idx", 0.0),
                float(attrs.get("node_type", 0.0)),
                float(attrs.get("power", 0.0)),
            ])
            pos[i] = torch.tensor([attrs.get("x", 0.0), attrs.get("y", 0.0)])

        src_list, dst_list, ef_list = [], [], []
        for u, v, edata in G.edges(data=True):
            iu, iv = node_idx[u], node_idx[v]
            feat = [
                edata.get("length",    0.0),
                edata.get("width",     0.0),
                edata.get("layer_idx", 0.0),
                edata.get("cut_area",  0.0),
                edata.get("num_cuts",  0.0),
                self._EDGE_TYPE_MAP.get(edata.get("edge_type", "wire"), 0.0),
            ]
            src_list += [iu, iv]; dst_list += [iv, iu]
            ef_list  += [feat, feat]

        # Normalize power (col 2) per-sample to [0, 1] so it matches the
        # scale of layer_idx and node_type; PDN nodes (power=0) stay at 0.
        pmax = x[:, 2].max()
        if pmax > 1e-12:
            x[:, 2] = x[:, 2] / pmax

        demand_mask  = (x[:, 1] > 0.5)  # node_type == 1
        tile_map = getattr(self, "demand_tile_map", {})
        # Only nodes that are actual demand nodes (node_type=1) and have tile mappings
        self.demand_node_order = [
            n for n in nodes
            if n in tile_map and G.nodes[n].get("node_type", 0) == 1.0
        ]

        return Data(
            x=x,
            edge_index=torch.tensor([src_list, dst_list], dtype=torch.long),
            edge_attr=torch.tensor(ef_list, dtype=torch.float32),
            pos=pos,
            demand_mask=demand_mask,
        )

    def _snap_node(self, x, y, layer):
        s = self.grid_snap
        return (round(x / s) * s, round(y / s) * s, layer)

    def _add_segment_edges(self, G):
        for seg in (s for s in self.segments if s.net_name == self.target_net):
            layer = self.layers.get(seg.layer)
            if layer is None: continue
            n1 = self._snap_node(seg.x1, seg.y1, seg.layer)
            n2 = self._snap_node(seg.x2, seg.y2, seg.layer)
            length    = math.hypot(seg.x2 - seg.x1, seg.y2 - seg.y1)
            width     = seg.width if seg.width > 0 else layer.width
            layer_idx = self._layer_order.get(seg.layer, 0)
            for node, (cx, cy) in [(n1, (seg.x1, seg.y1)), (n2, (seg.x2, seg.y2))]:
                if node not in G:
                    G.add_node(node, x=cx, y=cy, layer_idx=layer_idx,
                               is_seg_endpoint=True)
            if n1 == n2:
                continue
            if not G.has_edge(n1, n2):
                G.add_edge(n1, n2, length=length, width=width, layer_idx=float(layer_idx),
                           cut_area=0.0, num_cuts=0.0, edge_type="wire")

    def _build_wire_cache(self, G):
        h_cache = {}  
        v_cache = {}  
        for u, v, edata in G.edges(data=True):
            if edata.get("edge_type") != "wire": continue
            ux, uy = G.nodes[u]["x"], G.nodes[u]["y"]
            vx, vy = G.nodes[v]["x"], G.nodes[v]["y"]
            li = G.nodes[u].get("layer_idx", 0)
            w  = edata.get("width", 0.0)
            if abs(uy - vy) < 1e-6:
                h_cache.setdefault((li, round(uy, 4)), []).append(
                    (min(ux, vx), max(ux, vx), u, v, w))
            elif abs(ux - vx) < 1e-6:
                v_cache.setdefault((li, round(ux, 4)), []).append(
                    (min(uy, vy), max(uy, vy), u, v, w))
        return h_cache, v_cache

    def _insert_into_wire(self, G, node, x, y, layer_idx, h_cache, v_cache):
        for cache, fixed_key, coord, get_len in [
            (h_cache, (layer_idx, round(y, 4)), x, lambda na, nb: abs(G.nodes[na]["x"] - G.nodes[nb]["x"])),
            (v_cache, (layer_idx, round(x, 4)), y, lambda na, nb: abs(G.nodes[na]["y"] - G.nodes[nb]["y"])),
        ]:
            if fixed_key not in cache: continue
            entries = cache[fixed_key]
            for i, entry in enumerate(entries):
                lo, hi, nl, nr, w = entry
                if lo + 1e-6 < coord < hi - 1e-6:
                    if G.has_edge(nl, nr):
                        G.remove_edge(nl, nr)
                    for na, nb in [(nl, node), (node, nr)]:
                        G.add_edge(na, nb, length=get_len(na, nb), width=w,
                                   layer_idx=float(layer_idx), cut_area=0.0,
                                   num_cuts=0.0, edge_type="wire")
                    entries[i] = entries[-1]
                    entries[-1] = (lo, coord, nl, node, w)
                    entries.append((coord, hi, node, nr, w))
                    return

    def _add_via_edges(self, G):
        h_cache, v_cache = self._build_wire_cache(G)
        for via in (v for v in self.vias if v.net_name == self.target_net):
            if not via.bot_layer or not via.top_layer: continue
            nb = self._snap_node(via.x, via.y, via.bot_layer)
            nt = self._snap_node(via.x, via.y, via.top_layer)
            bot_idx = self._layer_order.get(via.bot_layer, 0)
            top_idx = self._layer_order.get(via.top_layer, 0)
            for node, li in [(nb, bot_idx), (nt, top_idx)]:
                if not G.has_node(node):
                    G.add_node(node, x=via.x, y=via.y, layer_idx=li)
            if not G.has_edge(nb, nt):
                G.add_edge(nb, nt,
                           length=0.0, width=0.0, layer_idx=float(bot_idx),
                           cut_area=float(via.cut_area), num_cuts=float(via.num_cuts),
                           edge_type="via")
            self._insert_into_wire(G, nb, via.x, via.y, bot_idx, h_cache, v_cache)
            self._insert_into_wire(G, nt, via.x, via.y, top_idx, h_cache, v_cache)

    def _add_demand_node(self, G, power):
        T, ny, nx      = power.shape
        xl, yl, xh, yh = self.die_area
        ts             = self.TILE_SIZE
        m1_idx         = self._layer_order.get(self.m1_layer, 0)

        
        m1_nodes_by_rail = {} 
        for n in G.nodes():
            if G.nodes[n].get("layer_idx") != m1_idx: continue
            if G.nodes[n].get("node_type", 0) == 1.0: continue  
            ny_ = round(G.nodes[n]["y"], 4)
            
            w = next((G[n][nb].get("width", 0.0)
                      for nb in G.neighbors(n)
                      if G[n][nb].get("edge_type") == "wire"), 0.0)
            m1_nodes_by_rail.setdefault(ny_, []).append((G.nodes[n]["x"], n, w))

        if not m1_nodes_by_rail:
            return
        sorted_rails = sorted(m1_nodes_by_rail.keys())

        # Maps cell_node → list of (row, col) tiles that snapped to it.
        # Used by load_sample to aggregate IR drop labels consistently.
        self.demand_tile_map: dict = {}

        for row in range(ny):
            for col in range(nx):
                p = power[:, row, col]
                if not np.any(p): continue

                cx = xl + (col + 0.5) * ts
                cy = yl + (row + 0.5) * ts

                # bisect to find nearest rail_y in O(log R)
                i = bisect.bisect_left(sorted_rails, cy)
                if i == 0:
                    rail_y = sorted_rails[0]
                elif i == len(sorted_rails):
                    rail_y = sorted_rails[-1]
                else:
                    a, b = sorted_rails[i - 1], sorted_rails[i]
                    rail_y = a if abs(a - cy) <= abs(b - cy) else b
                candidates = m1_nodes_by_rail[rail_y]
                if not candidates: continue

                nearest_x, nearest_n, w = min(candidates, key=lambda t: abs(t[0] - cx))
                length = abs(cx - nearest_x)

                cell_node = self._snap_node(cx, rail_y, self.m1_layer)
                self.demand_tile_map.setdefault(cell_node, []).append((row, col))
                if cell_node in G:
                    G.nodes[cell_node]["power"] = max(
                        G.nodes[cell_node].get("power", 0.0), float(p.max()))
                    continue
                G.add_node(cell_node, x=cx, y=rail_y,
                           layer_idx=self._tile_layer_idx,
                           power=float(p.max()), node_type=1.0)
                G.add_edge(cell_node, nearest_n,
                           length=length, width=w,
                           layer_idx=float(m1_idx), cut_area=0.0, num_cuts=0.0,
                           edge_type="inject")


    def _prune_seg_endpoints(self, G):
        """Remove segment endpoint nodes that ended up as dead ends (degree=1)."""
        to_remove = [
            n for n in G.nodes()
            if G.nodes[n].get("is_seg_endpoint") and G.degree(n) == 1
        ]
        G.remove_nodes_from(to_remove)

    def _collapse_via_stacks(self, G):
        """Remove pure via pass-through nodes (degree=2, only via edges).
        Collapse each chain into one via edge: num_cuts=min, cut_area=mean."""
        def is_passthru(n):
            return (G.degree(n) == 2 and
                    all(G[n][nb].get("edge_type") == "via" for nb in G.neighbors(n)))

        visited = set()
        to_remove = []

        for start in list(G.nodes()):
            if not is_passthru(start) or start in visited:
                continue

            # walk the chain in both directions to find endpoints
            chain_nodes = [start]
            chain_edges = []  # via edge data along the chain
            visited.add(start)

            start_neighbors = list(G.neighbors(start))
            for k, direction in enumerate(start_neighbors):
                cur, prev = direction, start
                while is_passthru(cur) and cur not in visited:
                    visited.add(cur)
                    chain_nodes.append(cur)
                    edata = G[prev][cur]
                    chain_edges.append(edata)
                    nxt = next(nb for nb in G.neighbors(cur) if nb != prev)
                    prev, cur = cur, nxt
                # cur is now the endpoint (non-passthru)
                chain_edges.append(G[prev][cur])
                if k == 0:
                    ep1 = cur
                else:
                    ep2 = cur

            # aggregate via properties across the chain
            all_edges = [G[start][nb] for nb in G.neighbors(start)] + chain_edges
            num_cuts = min(e.get("num_cuts", 1.0) for e in all_edges if e.get("edge_type")=="via")
            cut_areas = [e.get("cut_area", 0.0) for e in all_edges
                         if e.get("edge_type")=="via" and e.get("cut_area",0.0)>0]
            cut_area  = float(np.mean(cut_areas)) if cut_areas else 0.0
            bot_idx   = float(min(G.nodes[n].get("layer_idx",0) for n in chain_nodes))

            if not G.has_edge(ep1, ep2):
                G.add_edge(ep1, ep2,
                           length=0.0, width=0.0, layer_idx=bot_idx,
                           cut_area=cut_area, num_cuts=num_cuts,
                           edge_type="via")
            to_remove.extend(chain_nodes)

        G.remove_nodes_from(to_remove)

    def _collapse_wire_chains(self, G):
        def is_wire_passthru(n):
            if G.nodes[n].get("node_type", 0) == 1.0: return False
            if G.degree(n) != 2: return False
            return all(G[n][nb].get("edge_type") == "wire" for nb in G.neighbors(n))

        visited = set()
        to_remove = []

        for start in list(G.nodes()):
            if not is_wire_passthru(start) or start in visited:
                continue

            # walk both directions to find chain endpoints
            chain_nodes = [start]
            visited.add(start)
            endpoints = []

            for direction in list(G.neighbors(start)):
                cur, prev = direction, start
                total_length = G[prev][cur].get("length", 0.0)
                width        = G[prev][cur].get("width",  0.0)
                layer_idx    = G[prev][cur].get("layer_idx", 0.0)

                while is_wire_passthru(cur) and cur not in visited:
                    visited.add(cur)
                    chain_nodes.append(cur)
                    nxt = next(nb for nb in G.neighbors(cur) if nb != prev)
                    total_length += G[cur][nxt].get("length", 0.0)
                    prev, cur = cur, nxt

                endpoints.append((cur, total_length, width, layer_idx))

            if len(endpoints) == 2:
                ep1, l1, w1, li1 = endpoints[0]
                ep2, l2, _,  _   = endpoints[1]
                if not G.has_edge(ep1, ep2):
                    G.add_edge(ep1, ep2,
                               length=l1 + l2, width=w1,
                               layer_idx=li1, cut_area=0.0,
                               num_cuts=0.0, edge_type="wire")
            to_remove.extend(chain_nodes)

        G.remove_nodes_from(to_remove)

    def _compute_node_features(self, G):
        n_layers = len(self._layer_order) + 1
        for node in G.nodes():
            attrs = G.nodes[node]
            attrs["layer_idx"] = attrs.get("layer_idx", 0) / n_layers
