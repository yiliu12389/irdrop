import torch

def graph_smote(h, label, k=5, ratio=1.0):
    device = h.device
    minority_mask = (label == 1)
    h_min = h[minority_mask]

    if h_min.shape[0] < 2:
        return h, label
    k = min(k, h_min.shape[0] - 1)  # 热点数不足 k 时收缩

    M_min = h_min.shape[0]
    dist = torch.cdist(h_min, h_min, p=2)
    dist.fill_diagonal_(1e9)
    knn_idx = dist.topk(k, largest=False).indices  # (M_min, k)
    num_new = int(M_min * ratio)
    synthetic = []
    idx_pairs = []

    for _ in range(num_new):
        i = torch.randint(0, M_min, (1,)).item()
        nn = knn_idx[i][torch.randint(0, k, (1,)).item()]
        h_i  = h_min[i]
        h_j  = h_min[nn]
        lam = torch.rand(1, device=device)
        h_new = (1 - lam) * h_i + lam * h_j
        synthetic.append(h_new)
        idx_pairs.append((i, nn))

    synthetic = torch.stack(synthetic, dim=0)
    h_aug = torch.cat([h, synthetic], dim=0)
    label_new = torch.ones(num_new, device=device)
    label_aug = torch.cat([label, label_new], dim=0)

    return h_aug, label_aug


def build_augmented_adj(h_aug, num_orig, edge_index_orig, k=5):
    """
    Build edge index for the augmented graph (original + synthetic nodes).

    Original nodes keep their edges from edge_index_orig (tile-remapped).
    Synthetic nodes (index num_orig..) are connected to their k nearest
    original nodes via k-NN on embeddings.

    h_aug:          (N_orig + N_syn, hidden)
    num_orig:       number of original tile nodes
    edge_index_orig: (2, E) edge index already remapped to tile-node indices
    """
    N_syn  = h_aug.size(0) - num_orig
    device = h_aug.device

    if N_syn <= 0:
        return edge_index_orig

    h_syn = h_aug[num_orig:]   # (N_syn, hidden)
    h_ori = h_aug[:num_orig]   # (N_orig, hidden)

    k = min(k, num_orig)
    with torch.no_grad():
        dists   = torch.cdist(h_syn, h_ori, p=2)       # (N_syn, N_orig)
        knn_idx = dists.topk(k, largest=False).indices  # (N_syn, k)

    src = (torch.arange(N_syn, device=device)
           .unsqueeze(1).expand(-1, k).reshape(-1) + num_orig)
    dst = knn_idx.reshape(-1)
    syn_edges = torch.stack(
        [torch.cat([src, dst]),
         torch.cat([dst, src])], dim=0
    )

    return torch.cat([edge_index_orig, syn_edges], dim=1)
