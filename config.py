import re
import os
from pathlib import Path

import torch

BASE_DIR    = Path(__file__).parent
LEF_PATH    = BASE_DIR / "circuitnet.lef"
DEF_DIR     = BASE_DIR / "data" / "DEF"
POWER_DIR   = BASE_DIR / "data" / "power_t"
LABEL_DIR   = BASE_DIR / "data" / "IR_drop"
CKPT_PATH   = BASE_DIR / "irdrop_model.pt"

POWER_NETS  = ["VDD", "VSS"]
TARGET_NET  = "VDD"

HIDDEN      = 64
N_LAYERS    = 6
NODE_DIM    = 3
DROPOUT     = 0.1
LR          = 1e-3
EPOCHS       = 20
WEIGHT_DECAY = 1e-4
VAL_RATIO   = 0.15
SPLIT_SEED  = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
HOT_THRESH  = 0.040

TRAIN_DESIGNS = {"RISCY-a", "RISCY-b", "RISCY-FPU-a", "RISCY-FPU-b"}
TEST_DESIGNS  = {"zero-riscy-a", "zero-riscy-b"}


_DESIGN_RE = re.compile(r"^(RISCY-FPU-[ab]|RISCY-[ab]|zero-riscy-[ab])")

_DEF_KEYWORDS = {
    "NEW", "ROUTED", "FIXED", "COVER", "SHAPE", "RING", "STRIPE",
    "FOLLOWPIN", "IOWIRE", "COREWIRE", "BLOCKWIRE", "FILLWIRE",
    "NOSHIELD", "PLACED", "UNPLACED", "SOURCE", "SPECIAL", "END",
    "NETS", "COMPONENTS", "PINS", "VIAS", "NONDEFAULTRULES",
}