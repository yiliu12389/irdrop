"""LEF/DEF parsers and RC/L physical tables."""

import re
import gzip
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from config import _DEF_KEYWORDS


# ─── LEF dataclasses ─────────────────────────────────────────────────────────

@dataclass
class LEFLayer:
    name:      str
    direction: str
    pitch_x:   float
    pitch_y:   float
    width:     float
    height:    float = 0.0
    offset_x:  float = 0.0
    offset_y:  float = 0.0
    thickness: float = 0.0

@dataclass
class Rect:
    xl: float; yl: float; xh: float; yh: float

@dataclass
class MacroPin:
    name:        str
    direction:   str = ""
    use:         str = ""
    layer_rects: Dict[str, List[Rect]] = field(default_factory=dict)
    def is_supply(self): return self.use in ("POWER", "GROUND")

@dataclass
class LEFMacro:
    name:   str
    cls:    str   = ""
    size_x: float = 0.0
    size_y: float = 0.0
    site:   str   = ""
    pins:   Dict[str, MacroPin] = field(default_factory=dict)


class LEFParser:
    def __init__(self, lef_path: str):
        self.lef_path = Path(lef_path)
        self.layers: Dict[str, LEFLayer] = {}
        self.macros: Dict[str, LEFMacro] = {}

    def parse(self):
        if not self.lef_path.exists():
            warnings.warn(f"LEF not found: {self.lef_path}")
            return {}, {}
        text = self.lef_path.read_text(encoding="utf-8", errors="ignore")
        self._parse_layers(text)
        self._parse_macros(text)
        return self.layers, self.macros

    def _parse_layers(self, text):
        for m in re.finditer(r"^LAYER\s+(\w+)\s*\n(.*?)^END\s+\1", text, re.M | re.S):
            name, body = m.group(1), m.group(2)
            layer = LEFLayer(name, direction="", pitch_x=0.0, pitch_y=0.0, width=0.0)
            layer.direction = self._get(body, r"DIRECTION\s+(\w+)")
            layer.width     = self._float(body, r"(?<![A-Z])WIDTH\s+([\d.]+)")
            layer.thickness = self._float(body, r"THICKNESS\s+([\d.]+)")
            layer.height    = self._float(body, r"HEIGHT\s+([\d.]+)")
            pm = re.search(r"PITCH\s+([\d.]+)(?:\s+([\d.]+))?", body)
            if pm:
                layer.pitch_x = float(pm.group(1))
                layer.pitch_y = float(pm.group(2)) if pm.group(2) else layer.pitch_x
            om = re.search(r"OFFSET\s+([\d.\-]+)(?:\s+([\d.\-]+))?", body)
            if om:
                layer.offset_x = float(om.group(1))
                layer.offset_y = float(om.group(2)) if om.group(2) else layer.offset_x
            self.layers[name] = layer

    def _parse_macros(self, text):
        for m in re.finditer(r"^MACRO\s+(\w+)\s*\n(.*?)^END\s+\1", text, re.M | re.S):
            name, body = m.group(1), m.group(2)
            macro = LEFMacro(name=name, cls=self._get(body, r"CLASS\s+(\w+(?:\s+\w+)?)\s*;"))
            sm = re.search(r"SIZE\s+([\d.]+)\s+BY\s+([\d.]+)", body)
            if sm:
                macro.size_x, macro.size_y = float(sm.group(1)), float(sm.group(2))
            for pm in re.finditer(r"^\s*PIN\s+(\w+)\s*\n(.*?)^\s*END\s+\1", body, re.M | re.S):
                pin = MacroPin(
                    name=pm.group(1),
                    direction=self._get(pm.group(2), r"DIRECTION\s+(\w+)"),
                    use=self._get(pm.group(2), r"USE\s+(\w+)"),
                )
                self._parse_rects(pm.group(2), pin.layer_rects)
                macro.pins[pin.name] = pin
            self.macros[name] = macro

    def _parse_rects(self, body, store):
        cur = None
        for line in body.splitlines():
            line = line.strip().rstrip(";")
            lm = re.match(r"LAYER\s+(\w+)", line)
            if lm: cur = lm.group(1); continue
            rm = re.match(r"RECT\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)", line)
            if rm and cur:
                store.setdefault(cur, []).append(Rect(*map(float, rm.groups())))

    @staticmethod
    def _get(text, pat, default=""):
        m = re.search(pat, text)
        return m.group(1).strip() if m else default

    @staticmethod
    def _float(text, pat):
        m = re.search(pat, text)
        return float(m.group(1)) if m else 0.0


# ─── DEF dataclasses ─────────────────────────────────────────────────────────

@dataclass
class ViaRule:
    name: str; bot_layer: str; cut_layer: str; top_layer: str; resistance: float

@dataclass
class PDNSegment:
    net_name: str; layer: str
    x1: float; y1: float; x2: float; y2: float; width: float

@dataclass
class PDNVia:
    net_name: str; via_name: str; x: float; y: float
    bot_layer: str = ""; top_layer: str = ""; num_cuts: int = 1; cut_area: float = 0.0


def _read_def_text(def_path: Path) -> str:
    if def_path.suffix == ".gz":
        with gzip.open(def_path, "rt", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return def_path.read_text(encoding="utf-8", errors="ignore")


class DEFParser:
    def __init__(self, def_path: str, power_nets: List[str], dbu: int = 2000):
        self.def_path   = Path(def_path)
        self.power_nets = set(power_nets)
        self.dbu        = dbu
        self.segments:  List[PDNSegment] = []
        self.vias:      List[PDNVia]     = []
        self.die_area:  Tuple = (0, 0, 0, 0)
        self.via_info:  Dict  = {}

    def parse(self):
        if not self.def_path.exists():
            warnings.warn(f"DEF not found: {self.def_path}")
            return [], []
        text = _read_def_text(self.def_path)
        self._parse_units(text)
        self._parse_die_area(text)
        self._parse_via_defs(text)
        self._parse_specialnets(text)
        return self.segments, self.vias

    def _parse_units(self, text):
        m = re.search(r"UNITS\s+DISTANCE\s+MICRONS\s+(\d+)", text)
        if m: self.dbu = int(m.group(1))

    def _parse_die_area(self, text):
        m = re.search(
            r"DIEAREA\s*\(\s*([\d\-]+)\s+([\d\-]+)\s*\)\s*\(\s*([\d\-]+)\s+([\d\-]+)\s*\)", text
        )
        if m:
            self.die_area = tuple(int(v) / self.dbu for v in m.groups())

    def _parse_via_defs(self, text):
        m = re.search(r"\bVIAS\b\s+\d+\s*;(.*?)END\s+VIAS", text, re.S)
        if not m: return
        current_name = None
        current = {}

        def _flush(name, d):
            la = d.get("la", ""); lb = d.get("lb", "")
            try:
                lo, hi = sorted([la, lb], key=lambda s: int(re.search(r'\d+', s).group()))
            except Exception:
                lo, hi = la, lb
            num_cuts = d.get("rows", 1) * d.get("cols", 1)
            cut_area = (d.get("cut_w", 0.0) / self.dbu) * (d.get("cut_h", 0.0) / self.dbu)
            self.via_info[name] = (lo, hi, num_cuts, cut_area)

        for line in m.group(1).splitlines():
            nm = re.match(r"\s*-\s+(\S+)", line)
            if nm:
                if current_name: _flush(current_name, current)
                current_name = nm.group(1); current = {}; continue
            if not current_name: continue
            lm = re.search(r"\+\s+LAYERS\s+(\w+)\s+\w+\s+(\w+)", line)
            if lm: current["la"], current["lb"] = lm.group(1), lm.group(2)
            cm = re.search(r"\+\s+CUTSIZE\s+([\d.]+)\s+([\d.]+)", line)
            if cm: current["cut_w"] = float(cm.group(1)); current["cut_h"] = float(cm.group(2))
            rm = re.search(r"\+\s+ROWCOL\s+(\d+)\s+(\d+)", line)
            if rm: current["rows"] = int(rm.group(1)); current["cols"] = int(rm.group(2))
        if current_name: _flush(current_name, current)

    def _parse_specialnets(self, text):
        m = re.search(r"SPECIALNETS\s+\d+\s*;(.*?)END\s+SPECIALNETS", text, re.S)
        if not m: return
        for net_block in re.split(r"\n\s*-\s+", m.group(1))[1:]:
            nm = re.match(r"(\w+)", net_block.strip())
            if nm and nm.group(1) in self.power_nets:
                self._parse_net_block(nm.group(1), net_block)

    def _parse_net_block(self, net_name, block):
        current_layer = None; current_width = 0.0
        for sub in re.split(r"\bNEW\s+", block):
            lm = re.search(r"(?:ROUTED|FIXED|COVER)\s+(\w+)\s+([\d.]+)", sub)
            if lm:
                current_layer = lm.group(1)
                current_width = float(lm.group(2)) / self.dbu
            else:
                # NEW sub-block: layer and width at the start, e.g. "M1 300 + SHAPE ..."
                nm = re.match(r"\s*(\w+)\s+([\d.]+)", sub)
                if nm:
                    current_layer = nm.group(1)
                    current_width = float(nm.group(2)) / self.dbu
            if current_layer is None: continue
            coords_raw = re.findall(r"\(\s*([\d\-*]+)\s+([\d\-*]+)\s*\)", sub)
            via_names  = [
                w for w in re.findall(r"\)\s*([A-Za-z]\w*)", sub)
                if w.upper() not in _DEF_KEYWORDS
            ]
            prev = None
            for i, (sx, sy) in enumerate(coords_raw):
                x = prev[0] if sx == '*' else int(sx) / self.dbu
                y = prev[1] if sy == '*' else int(sy) / self.dbu
                if prev is not None and current_width > 0:
                    px, py = prev
                    if not (x == px and y == py):
                        self.segments.append(PDNSegment(
                            net_name=net_name, layer=current_layer,
                            x1=px, y1=py, x2=x, y2=y, width=current_width
                        ))
                if i < len(via_names):
                    vname = via_names[i]
                    info  = self.via_info.get(vname)
                    self.vias.append(PDNVia(
                        net_name=net_name, via_name=vname, x=x, y=y,
                        bot_layer=info[0] if info else "",
                        top_layer=info[1] if info else "",
                        num_cuts=info[2] if info else 1,
                        cut_area=info[3] if info else 0.0,
                    ))
                prev = (x, y)


