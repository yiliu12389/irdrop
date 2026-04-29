import os
import re
import numpy as np
import torch
import os
import pandas as pd
from torch_geometric.data import Data
import gzip

def process_pdn():
    data_root = r"C:\Users\s1155250660\Desktop\irdrop\data"
    def_root  = r"C:\Users\s1155250660\Desktop\irdrop\data\DEF"
    output_dir = r"C:\Users\s1155250660\Desktop\irdrop\graph"
    
    os.makedirs(output_dir, exist_ok=True)


    power_all_dir = os.path.join(data_root, "power_all")
    if not os.path.exists(power_all_dir):
        print(f"{power_all_dir}")
        return

    samples = [d for d in os.listdir(power_all_dir)]

    for sample_name in samples:
        pure_name = sample_name.replace('.npy', '')
        print(f"Processing: {pure_name}")
        
        def_path = os.path.join(def_root, f"{pure_name}.def.gz")
        
        if not os.path.exists(def_path):
            possible_defs = [f for f in os.listdir(def_root) if sample_name in f and f.endswith('.def')]
            if possible_defs:
                def_path = os.path.join(def_root, possible_defs[0])
            else:
                print(f"error  {sample_name}.def")
                continue


        try:
            data_obj = graph_data(data_root, def_path, pure_name)
            
            save_file = os.path.join(output_dir, f"{sample_name}.pt")
            torch.save(data_obj, save_file)
            print(f"  success {save_file}")
            

            del data_obj

        except Exception as e:
            print(f"  error {str(e)}")

def graph_data(data_dir, def_path, sample_name):
    path_all = os.path.join(data_dir, "power_all", sample_name)
    path_sca = os.path.join(data_dir, "power_sca", sample_name)
    path_t   = os.path.join(data_dir, "power_t",   sample_name)
    
    power_all = np.load(path_all)
    power_sca = np.load(path_sca)
    power_t   = np.load(path_t)
    
    L = 4500 
    H,W = power_all.shape
    N = H * W
    max_l = 6
    re_max_l = re.compile(r'M(\d+)')

    open_func = gzip.open if def_path.endswith('.gz') else open
    mode = 'rt' if def_path.endswith('.gz') else 'r'

    with open_func(def_path, mode, encoding='utf-8') as f:
        for line in f:
            if "ROUTED" in line or "NEW" in line:
                m = re_max_l.search(line)
                if m:
                    K = int(m.group(1))
                    break
             
    vias_dict = {}    # {name: [cuts, layer_idx]}
    edge_weights = {} # {(u, v): np.array([k_channels])}
    v_slots = {}      # {(r, c, l_idx): total_cuts}


    re_pdn = re.compile(r'(?:ROUTED|NEW)\s+M(\d+)\s+(\d+)')
    re_pt  = re.compile(r'\(\s*(\d+|\*)\s+(\d+|\*)\s*\)(?:\s+([A-Z0-9_]+))?')

    with open_func(def_path, mode, encoding='utf-8') as f:
        curr_l, curr_w, v_def_name = 0, 0, None  
        mode = "IDLE" 
        
        for line in f:
            line_s = line.strip()
            if not line_s: continue

            if line_s.startswith("VIAS"): mode = "VIAS"; continue
            if line_s.startswith("SPECIALNETS"): mode = "SPECIAL"; continue
            if line_s.startswith("END VIAS"): mode = "IDLE"; continue
            if line_s.startswith("END SPECIALNETS"): mode = "IDLE"; continue
                
            if mode == "VIAS":
                if line_s.startswith("-"):
                    v_match = re.search(r'-\s+([A-Za-z0-9_]+)',line_s)
                    if v_match:
                        curr_via = v_match.group(1)
                        l_match = re.search(r'\d+', curr_via)
                        l_val = int(l_match.group()[0])
                        vias_dict[curr_via] = [l_val - 1,0,0,1,1]
                if "CUTSPACING" in line_s:
                    sp_match = re.search(r'CUTSPACING\s+(\d+)\s+(\d+)',line_s)
                    if sp_match:
                        vias_dict[curr_via][2] = int(sp_match.group(1))
                        vias_dict[curr_via][3] = int(sp_match.group(2))
            
                if "ROWCOL" in line_s and curr_via:
                    rc_match = re.search(r'ROWCOL\s+(\d+)\s+(\d+)',line_s)
                    if rc_match:
                        vias_dict[curr_via][3] = int(rc_match.group(1)) 
                        vias_dict[curr_via][4] = int(rc_match.group(2)) 
                        
            elif mode == "SPECIAL":
                pdn_match = re_pdn.search(line)
                if pdn_match:
                        layer = int(pdn_match.group(1)) - 1
                        width = float(pdn_match.group(2)) / L
                if "(" in line_s:
                    matches = re_pt.findall(line_s)
                    prev_x, prev_y = None, None
                    for cx, cy, via_name in matches:
                        curr_x = prev_x if cx == '*' else int(cx)
                        curr_y = prev_y if cy == '*' else int(cy)
                    
                        if via_name in vias_dict:
                            l_idx, dx, dy, rows, cols = vias_dict[via_name]
                            x_max = curr_x + (cols - 1) * dx
                            y_max = curr_y + (rows - 1) * dy
                            c_start, c_end = curr_x // L, x_max // L
                            r_start, r_end = curr_y // L, y_max // L
                            num_grids = (r_end - r_start + 1) * (c_end - c_start + 1)
                            cuts_per_grid = rows * cols / num_grids
                            for rg in range(int(r_start), int(r_end) + 1):
                                    for cg in range(int(c_start), int(c_end) + 1):
                                        tr, tc = min(rg, H - 1), min(cg, W - 1)
                                        v_slots[(tr, tc, l_idx)] = v_slots.get((tr, tc, l_idx), 0) + cuts_per_grid

                        if prev_x is not None :
                            r0, c0 = min(prev_y // L, H - 1), min(prev_x // L, W - 1)
                            r, c = min(curr_y // L, W - 1), min(curr_x // L, H - 1)
                            for i in range(min(r0, r), max(r0, r) + 1):
                                for j in range(min(c0, c), max(c0, c) + 1):
                                    for dr, dc in [(0, 1), (1, 0)]:
                                        ni, nj = i + dr, j + dc
                                        if ni <= max(r0, r) and nj <= max(c0, c) and ni < H and nj < W:
                                            u, v = i * W + j, ni * W + nj
                                            edge = tuple(sorted((u, v)))
                                            if edge not in edge_weights: 
                                                edge_weights[edge] = {}
                                            edge_weights[edge][layer] = edge_weights[edge].get(layer, 0) + width
                                    
                        prev_x, prev_y = curr_x, curr_y
 
    via_tensor = np.zeros((N, K), dtype=np.float32)
    for (r, c, l), cuts in v_slots.items():
        via_tensor[r * W + c, l] = cuts
    src, dst, attr = [], [], []
    for (u, v), w_dict in edge_weights.items():
        vec = np.zeros(K, dtype=np.float32)
        for l, w in w_dict.items(): vec[l] = w
        src.extend([u, v]); dst.extend([v, u])
        attr.extend([vec, vec])
    
    f_all = power_all.reshape(N,1)
    f_sca = power_sca.reshape(N,1)
    f_t = np.transpose(power_t, (1, 2, 0)).reshape(N, 20) if power_t.ndim == 3 else p_t.reshape(N, 20)
    
    x = torch.tensor (np.concatenate([f_all, f_sca, f_t, via_tensor], axis=1),dtype=torch.float)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(np.array(attr), dtype=torch.float)
    graph_data = Data(x=x, K=K, edge_index=edge_index, edge_attr=edge_attr)
    return graph_data


def generate_index_csv(pt_dir, label_dir, output_csv):
    data_list = []
    
    pt_files = [f for f in os.listdir(pt_dir) if f.endswith('.pt')]
    
    for f in pt_files:
        sample_name = f.replace('.pt', '')
        
        # 对应标签的文件名（假设标签是 .npy 格式）
        label_file = f"{sample_name}.npy" 
        label_path = os.path.join(label_dir, label_file)
        
        # 只有当特征和标签都存在时才记录
        if os.path.exists(label_path):
            data_list.append({
                'feature_path': os.path.join(pt_dir, f),
                'label_path': label_path,
                'sample_name': sample_name
            })
        else:
            print(f"Warning: 样本 {sample_name} 缺少对应的 Label 文件。")


    df = pd.DataFrame(data_list)
    df.to_csv(output_csv, index=False)
    print(f"索引表已生成: {output_csv}, 共计 {len(df)} 个对齐样本。")

if __name__ == "__main__":
    process_pdn()