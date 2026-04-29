import sys; sys.path.insert(0, 'e:/IR_drop')
import torch, time
from train_ode import prepare_sample, predict_demand_ir, load_lef
from model_ode import PDNQuasiStatic

device = torch.device('cpu')
lef    = load_lef()

print('[1] prepare sample...')
s = prepare_sample('826-RISCY-a-2-c2-u0.8-m4-p8-f0', lef, device)
print(f'    edge_attr={s["edge_attr"].shape}  N_w={s["N_w"]}')
print(f'    free={len(s["free_ids"])}  supply={len(s["supply_ids"])}')
print(f'    I_inj max={s["I_inj"].max():.3e}')
print(f'    ir_gt: max={s["ir_gt"].max()*1000:.1f}mV  n={len(s["ir_gt"])}')

model = PDNQuasiStatic(hidden=64, V_supply=0.85).to(device)

print()
print('[2] forward pass...')
t0 = time.time()
ir_free = model(s['edge_attr'], s['src'], s['dst'],
                s['N_w'], s['free_ids'], s['supply_ids'], s['I_inj'])
print(f'    time={time.time()-t0:.2f}s  ir_free shape={ir_free.shape}')
print(f'    IR drop  max={ir_free.max()*1000:.2f}mV  min={ir_free.min()*1000:.2f}mV')

pred = predict_demand_ir(ir_free, s['free_ids'], s['demand_wire'], s['N_w'])
print(f'    pred demand max={pred.max()*1000:.2f}mV')

loss = torch.nn.HuberLoss(delta=0.01)(pred, s['ir_gt'])
print(f'    loss={loss.item():.6f}')

print()
print('[3] backward pass...')
t0 = time.time()
loss.backward()
print(f'    time={time.time()-t0:.2f}s')
grad = model.R_mlp.net[0].weight.grad
print(f'    grad norm (layer0 weight)={grad.norm():.4f}')

print()
print('OK - gradient flows correctly')
