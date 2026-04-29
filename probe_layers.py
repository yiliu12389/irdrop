"""
逐层探针：在 826 样本上跑一次前向传播，
观察每层输出对热点 vs 非热点的分离程度。
"""
import sys, os, time
os.chdir('e:/IR_drop')
sys.path.insert(0, 'e:/IR_drop')

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from config import *
from dataset import load_sample, build_tile_edges
from model import IRDropGNN, EdgeGenerator, ClassifierGNN, PDNConv, VDD
# NODE_DIM=2: layer_idx, local_cap (norm_x/norm_y removed — design-specific)
from parsers import LEFParser

SAMPLE = '922-RISCY-a-2-c2-u0.85-m2-p3-f1'

# ── 加载样本 ──────────────────────────────────────────────────────────────────
print("加载样本...")
lef_data = LEFParser(str(LEF_PATH)).parse()
data, label, dt = load_sample(SAMPLE, lef_data)

x          = data.x[:, 2:4]
edge_index = data.edge_index
edge_attr  = data.edge_attr
power      = data.power
tile_mask  = (power.sum(dim=1) > 0)

print(f"  全图节点: {x.shape[0]:,}  tile节点: {tile_mask.sum().item():,}")
print(f"  热点: {int(label.sum())}  非热点: {int((label==0).sum())}")
print(f"  label shape: {label.shape}")

# ── 初始化模型（随机权重） ────────────────────────────────────────────────────
torch.manual_seed(42)
model = IRDropGNN(node_dim=NODE_DIM, edge_dim=3, hidden=HIDDEN,
                  n_layers=N_LAYERS, dropout=0.0)  # dropout=0 方便观察
model.eval()

hot_mask  = label.bool()   # 热点 tile
norm_mask = ~hot_mask      # 非热点 tile

def separation(h_tiles, hot_mask):
    """用 tile 嵌入的 L2 范数，看热点 vs 非热点的均值差距"""
    norms = h_tiles.norm(dim=-1)
    h_hot  = norms[hot_mask].mean().item()
    h_norm = norms[norm_mask].mean().item()
    try:
        auc = roc_auc_score(hot_mask.numpy().astype(int), norms.detach().numpy())
        auc = max(auc, 1.0 - auc)   # 取绝对方向（norm越小或越大均可）
    except Exception:
        auc = float('nan')
    return h_hot, h_norm, auc

# ── 手动逐层前向，记录中间状态 ────────────────────────────────────────────────
print("\n逐层传播中...\n")
print(f"{'层':<25} {'热点norm均值':>12} {'非热点norm均值':>14} {'AUC':>8} {'说明'}")
print("-" * 70)

with torch.no_grad():
    N, T = power.shape

    # ── node_enc（静态，power=0）────────────────────────────────────────────
    h = model.node_enc(torch.cat([x, torch.zeros(N, 1)], dim=-1))
    h_tiles = h[tile_mask]
    hh, hn, auc = separation(h_tiles, hot_mask)
    print(f"  {'node_enc (static)':<23} {hh:12.4f} {hn:14.4f} {auc:8.4f}  随机初始化基线")

    # ── static_convs ────────────────────────────────────────────────────────
    for i, conv in enumerate(model.static_convs):
        h = conv(h, edge_index, edge_attr)
        h_tiles = h[tile_mask]
        hh, hn, auc = separation(h_tiles, hot_mask)
        print(f"  {'static_conv['+str(i)+']':<23} {hh:12.4f} {hn:14.4f} {auc:8.4f}")

    h_static = h.clone()
    print()

    # ── GRU（新版：无 dynamic_conv）──────────────────────────────────────────
    h_t = h_static.clone()
    for t in range(T):
        p_t_vec = power[:, t] / VDD
        h_in = h_static + model.node_enc(torch.cat([x, p_t_vec.unsqueeze(-1)], dim=-1))
        h_t  = model.gru(h_in, h_t)

    h_tiles = h_t[tile_mask]
    hh, hn, auc = separation(h_tiles, hot_mask)
    print(f"  {'static_conv + GRU':<23} {hh:12.4f} {hn:14.4f} {auc:8.4f}  新版（无dynamic_conv）")

    # ── readout logits ───────────────────────────────────────────────────────
    logits = model.readout(h_t[tile_mask]).squeeze(-1)
    probs  = logits.sigmoid()
    try:
        auc_final = roc_auc_score(hot_mask.numpy().astype(int), probs.numpy())
    except Exception:
        auc_final = float('nan')
    print(f"  {'readout logits':<23} {'':>12} {'':>14} {auc_final:8.4f}  sigmoid 输出")

    print()
    print(f"  热点 logit:    mean={logits[hot_mask].mean():.3f}  "
          f"min={logits[hot_mask].min():.3f}  max={logits[hot_mask].max():.3f}")
    print(f"  非热点 logit:  mean={logits[norm_mask].mean():.3f}  "
          f"min={logits[norm_mask].min():.3f}  max={logits[norm_mask].max():.3f}")

# ── Linear Probe：逐层用 LogReg 拟合嵌入，看线性可分程度 ─────────────────────
print("\n\nLinear Probe (LogReg, 5-fold StratifiedCV, class_weight='balanced')")
print(f"  {'层':<25} {'mean AUC':>10} {'±std':>8}")
print("-" * 50)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

y_np = hot_mask.numpy().astype(int)

def linear_probe(h_np, y_np, n_splits=5):
    scaler = StandardScaler()
    h_s = scaler.fit_transform(h_np)
    clf = LogisticRegression(class_weight='balanced', max_iter=500, C=0.1, solver='lbfgs')
    cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    try:
        scores = cross_val_score(clf, h_s, y_np, cv=cv, scoring='roc_auc')
        return scores.mean(), scores.std()
    except Exception:
        return float('nan'), float('nan')

with torch.no_grad():
    N, T = power.shape
    layer_embeds = {}

    h = model.node_enc(torch.cat([x, torch.zeros(N, 1)], dim=-1))
    layer_embeds['node_enc'] = h[tile_mask].numpy()

    for i, conv in enumerate(model.static_convs):
        h = conv(h, edge_index, edge_attr)
        layer_embeds[f'static_conv[{i}]'] = h[tile_mask].numpy()

    h_static = h.clone()
    h_t = h_static.clone()
    for t in range(T):
        p_t = power[:, t] / VDD
        h_in = h_static + model.node_enc(torch.cat([x, p_t.unsqueeze(-1)], dim=-1))
        h_t = model.gru(h_in, h_t)
    layer_embeds['static_conv + GRU'] = h_t[tile_mask].numpy()

for name, emb in layer_embeds.items():
    mu, sd = linear_probe(emb, y_np)
    print(f"  {name:<25} {mu:10.4f} {sd:8.4f}")

# ── Shuffle Label 检验：打乱标签看 AUC 是否来自真实信号 ────────────────────────
print("\n\nShuffle Label Test (n=200 permutations, final GRU embedding)")
rng   = np.random.default_rng(0)
h_emb = layer_embeds['static_conv + GRU']   # (N_tile, hidden)
norms = np.linalg.norm(h_emb, axis=-1)

# 取绝对方向 AUC（与 separation() 一致）
raw_auc  = roc_auc_score(y_np, norms)
real_auc = max(raw_auc, 1.0 - raw_auc)
score    = norms if raw_auc >= 0.5 else -norms

null_aucs = []
for _ in range(500):
    y_shuf = rng.permutation(y_np)
    a = roc_auc_score(y_shuf, score)
    null_aucs.append(max(a, 1.0 - a))

null_aucs = np.array(null_aucs)
p_val     = (null_aucs >= real_auc).mean()
print(f"  真实 AUC  = {real_auc:.4f}")
print(f"  null mean = {null_aucs.mean():.4f}  std = {null_aucs.std():.4f}")
print(f"  p-value   = {p_val:.4f}  (>= real_auc 的置换比例)")
if p_val < 0.05:
    print("  结论: 显著 (p<0.05) → 嵌入携带真实热点信号，非随机偶然")
else:
    print("  结论: 不显著 → 当前分离可能是随机偶然")

# ── 可视化：热点 vs 非热点的嵌入分布 ─────────────────────────────────────────
with torch.no_grad():
    # 取前两个 PCA 主成分投影
    from sklearn.decomposition import PCA

    h_final = h_t[tile_mask].numpy()
    pca = PCA(n_components=2)
    h_2d = pca.fit_transform(h_final)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0d0d1a')

    # 左：PCA 投影
    ax = axes[0]
    ax.set_facecolor('#1a1a2e')
    ax.scatter(h_2d[~hot_mask.numpy(), 0], h_2d[~hot_mask.numpy(), 1],
               s=1, alpha=0.3, c='#457b9d', label=f'Normal ({(~hot_mask).sum()})')
    ax.scatter(h_2d[hot_mask.numpy(), 0], h_2d[hot_mask.numpy(), 1],
               s=40, alpha=1.0, c='#e63946', zorder=5, label=f'Hotspot ({hot_mask.sum()})')
    ax.set_title('GRU final embedding (PCA)', color='white')
    ax.legend(fontsize=8, facecolor='#0d0d1a', labelcolor='white')
    ax.tick_params(colors='gray')

    # 右：logit 分布
    ax = axes[1]
    ax.set_facecolor('#1a1a2e')
    ax.hist(logits[norm_mask].numpy(), bins=60, color='#457b9d',
            alpha=0.7, label='Normal', density=True)
    ax.hist(logits[hot_mask].numpy(), bins=15, color='#e63946',
            alpha=0.9, label='Hotspot', density=True)
    ax.axvline(0, color='yellow', linestyle='--', lw=1, label='threshold=0')
    ax.set_title('Logit distribution (random init)', color='white')
    ax.set_xlabel('logit', color='gray')
    ax.legend(fontsize=8, facecolor='#0d0d1a', labelcolor='white')
    ax.tick_params(colors='gray')

    plt.suptitle(f'826 sample — layer probe (random weights, AUC={auc_final:.3f})',
                 color='white', fontsize=11)
    plt.tight_layout()
    out = 'e:/IR_drop/layer_probe.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\n图表保存: {out}")
