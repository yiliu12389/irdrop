import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

DEF_PATH = Path("e:/IR_drop/826-RISCY-a-2-c2-u0.8-m4-p8-f0.def")
DATA_DIR = Path("e:/IR_drop/data")
SAMPLE   = "826-RISCY-a-2-c2-u0.8-m4-p8-f0"
HOT_THRESH = 40

# ── 加载数据 ──────────────────────────────────────────────────────────────────
label  = np.load(DATA_DIR / "IR_drop" / SAMPLE).astype(np.float32)
grid_h, grid_w = label.shape
die_x2, die_y2 = 1184400, 1182600
units  = 2000
cell_w = die_x2 / grid_w / units   # um per cell
cell_h = die_y2 / grid_h / units

# ── 解析 SPECIALNETS ──────────────────────────────────────────────────────────
segments = []
in_net   = False
cur_net  = cur_layer = None
cur_width = 0
last_x2 = last_y2 = 0

with open(DEF_PATH) as f:
    for line in f:
        line = line.strip()
        if re.match(r"^- (VDD|VSS)", line):
            m = re.match(r"^- (\w+)", line)
            cur_net = m.group(1)
            in_net  = True
            continue
        if in_net and line.startswith("END SPECIALNETS"):
            break
        if not in_net:
            continue
        m_l = re.search(r"(?:ROUTED|NEW)\s+(M\d+)\s+(\d+)", line)
        if m_l:
            cur_layer = m_l.group(1)
            cur_width = int(m_l.group(2))
        coords = re.findall(r"\(\s*(\*|\d+)\s+(\*|\d+)\s*\)", line)
        if len(coords) == 2 and cur_layer:
            (x1s, y1s), (x2s, y2s) = coords
            x1 = last_x2 if x1s == "*" else int(x1s)
            y1 = last_y2 if y1s == "*" else int(y1s)
            x2 = x1      if x2s == "*" else int(x2s)
            y2 = y1      if y2s == "*" else int(y2s)
            segments.append((cur_net, cur_layer, x1/units, y1/units,
                             x2/units, y2/units, cur_width/units))
            last_x2, last_y2 = x2, y2

# ── 热点坐标 (um) ─────────────────────────────────────────────────────────────
hot_coords = np.argwhere(label > HOT_THRESH)
hot_ir     = label[hot_coords[:, 0], hot_coords[:, 1]]
hot_x = (hot_coords[:, 1] + 0.5) * cell_w   # 中心 x
hot_y = (hot_coords[:, 0] + 0.5) * cell_h   # 中心 y

# ── 绘图区域：热点附近 ±80 um ─────────────────────────────────────────────────
cx   = hot_x.mean()
VIEW_X = (cx - 80,  cx + 80)     # x 窗口
VIEW_Y = (hot_y.min() - 20, hot_y.max() + 20)  # y 窗口

LAYER_STYLE = {
    "M8": dict(color="#e63946", alpha=0.55, zorder=4),
    "M7": dict(color="#457b9d", alpha=0.45, zorder=3),
    "M6": dict(color="#2a9d8f", alpha=0.40, zorder=3),
    "M5": dict(color="#e9c46a", alpha=0.35, zorder=2),
    "M4": dict(color="#f4a261", alpha=0.35, zorder=2),
    "M1": dict(color="#a8dadc", alpha=0.30, zorder=1),
}

fig, axes = plt.subplots(1, 2, figsize=(16, 10))

for ax_idx, (ax, view_x, view_y, title) in enumerate([
    (axes[0], VIEW_X, VIEW_Y,       "Hotspot Region (±80 um zoom)"),
    (axes[1], (0, die_x2/units), (0, die_y2/units), "Full-chip PDN + Hotspots"),
]):
    ax.set_xlim(*view_x)
    ax.set_ylim(*view_y)
    ax.set_facecolor("#1a1a2e")
    ax.set_xlabel("x (um)", fontsize=9)
    ax.set_ylabel("y (um)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_aspect("equal")

    # 画 VDD 线段
    drawn_layers = set()
    for (net, layer, x1, y1, x2, y2, w) in segments:
        if net != "VDD":
            continue
        style = LAYER_STYLE.get(layer, dict(color="white", alpha=0.2, zorder=1))
        rx1 = min(x1, x2) - w / 2
        ry1 = min(y1, y2) - w / 2
        rw  = abs(x2 - x1) + w
        rh  = abs(y2 - y1) + w
        # 裁剪到视图
        if rx1 > view_x[1] or rx1 + rw < view_x[0]:
            continue
        if ry1 > view_y[1] or ry1 + rh < view_y[0]:
            continue
        label_str = layer if layer not in drawn_layers else ""
        rect = patches.Rectangle(
            (rx1, ry1), rw, rh,
            linewidth=0,
            facecolor=style["color"],
            alpha=style["alpha"],
            zorder=style["zorder"],
            label=label_str if label_str else None,
        )
        ax.add_patch(rect)
        drawn_layers.add(layer)

    # 画热点（按 IR drop 着色）
    sc = ax.scatter(
        hot_x, hot_y,
        c=hot_ir, cmap="hot", vmin=40, vmax=hot_ir.max(),
        s=80 if ax_idx == 0 else 20,
        zorder=10,
        edgecolors="white", linewidths=0.5,
        label="Hotspot",
    )
    if ax_idx == 0:
        plt.colorbar(sc, ax=ax, label="IR drop (mV)", shrink=0.6)

        # 标出关键 M8 条带边界
        m8_x_right = 259.0
        ax.axvline(m8_x_right, color="#e63946", linestyle="--", lw=1.5,
                   label=f"M8 right edge x={m8_x_right:.1f}um", zorder=11)

        # 标出 M7 水平条带位置
        m7_ys = [62, 110.7, 159.3, 207.9, 256.5, 305.2, 353.8, 402.4, 451.1, 499.7]
        for my in m7_ys:
            if view_y[0] <= my <= view_y[1]:
                ax.axhline(my, color="#457b9d", linestyle=":", lw=0.8, alpha=0.7)

        # 标注最近热点
        worst_idx = np.argmax(hot_ir)
        ax.annotate(
            f"Max IR={hot_ir[worst_idx]:.0f}mV\n({hot_x[worst_idx]:.1f},{hot_y[worst_idx]:.1f})um",
            xy=(hot_x[worst_idx], hot_y[worst_idx]),
            xytext=(hot_x[worst_idx] + 25, hot_y[worst_idx] + 10),
            color="white", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="yellow", lw=1),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e63946", alpha=0.8),
        )

    ax.legend(loc="upper right", fontsize=7, framealpha=0.6,
              facecolor="#0d0d1a", labelcolor="white")
    ax.tick_params(colors="gray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

plt.suptitle(
    f"826-RISCY  IR Drop Hotspot vs PDN Structure\n"
    f"42 hotspot pixels  |  max={hot_ir.max():.0f} mV  |  all at col=115, x≈259 um",
    fontsize=11, color="white"
)
fig.patch.set_facecolor("#0d0d1a")
plt.tight_layout()

out = Path("e:/IR_drop/hotspot_pdn.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"saved → {out}")
