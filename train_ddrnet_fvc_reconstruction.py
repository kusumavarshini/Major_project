from pathlib import Path
import json, random, time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from DDRNet.models.DDRNet_DIR import DDRNet_DIR
from DDRNet.tools.fp_segmentation import segmentation_coherence

ROOT = Path(__file__).resolve().parent
LATENT_DIR = ROOT / 'data' / 'Latent_fvc'
CLEAN_DIR = ROOT / 'data' / 'clean_fvc_new'
OUTPUT_DIR = ROOT / 'experiments' / 'ddrnet_fvc_reconstruction'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = OUTPUT_DIR / 'best_model.pth'
HISTORY_PATH = OUTPUT_DIR / 'training_history.json'
SPLIT_PATH = OUTPUT_DIR / 'split.json'
SOURCE_SPLIT = ROOT / 'experiments' / 'ddrnet_fvc' / 'split.json'
INIT_CHECKPOINT = ROOT / 'experiments' / 'ddrnet_baseline' / 'best_model.pth'

# ---------------- CONFIG ----------------
SEED = 42
IMAGE_SIZE = 512
GRID_SIZE = 32
DIS_CONST = 16
BATCH_SIZE = 2
NUM_WORKERS = 0
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 30
PATIENCE = 12
RECON_WEIGHT = 1.0       # DDRNet warp -> clean
GRAD_WEIGHT = 0.5
SMOOTH_WEIGHT = 0.02
REFINE_WEIGHT = 1.0      # final reconstructed image -> clean
MAX_FIELD_MAG = 60.0


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FVCPairedDataset(Dataset):
    def __init__(self, ids):
        self.ids = list(ids)
        if not self.ids: raise RuntimeError('Dataset split is empty.')
        for i in self.ids:
            if not (LATENT_DIR/f'{i}.png').exists(): raise FileNotFoundError(LATENT_DIR/f'{i}.png')
            if not (CLEAN_DIR/f'{i}.png').exists(): raise FileNotFoundError(CLEAN_DIR/f'{i}.png')

    def __len__(self): return len(self.ids)

    def __getitem__(self, index):
        i = self.ids[index]
        latent = cv2.imread(str(LATENT_DIR/f'{i}.png'), cv2.IMREAD_GRAYSCALE)
        clean = cv2.imread(str(CLEAN_DIR/f'{i}.png'), cv2.IMREAD_GRAYSCALE)
        if latent is None or clean is None: raise RuntimeError(f'Could not read pair {i}')
        latent = cv2.resize(latent, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        clean = cv2.resize(clean, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        latent = (255.0 - latent.astype(np.float32))/255.0
        clean = (255.0 - clean.astype(np.float32))/255.0
        latent_u8 = np.clip(255.0 - latent*255.0, 0, 255).astype(np.uint8)
        mask = segmentation_coherence(latent_u8, win_size=16, stride=8)
        mask = np.asarray(mask, dtype=np.float32)
        if mask.shape != (GRID_SIZE, GRID_SIZE):
            mask = cv2.resize(mask, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)
        return {
            'latent': torch.from_numpy(latent).unsqueeze(0),
            'clean': torch.from_numpy(clean).unsqueeze(0),
            'mask': torch.from_numpy(mask).unsqueeze(0),
            'id': int(i),
        }


def warp_with_field(image, field):
    b, _, h, w = image.shape
    full = F.interpolate(field, size=(h,w), mode='bilinear', align_corners=True)
    dx, dy = full[:,0], full[:,1]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=image.dtype),
        torch.arange(w, device=image.device, dtype=image.dtype), indexing='ij')
    xx = xx.unsqueeze(0).expand(b,-1,-1); yy = yy.unsqueeze(0).expand(b,-1,-1)
    # IMPORTANT: matches corrected production inference convention.
    map_x = xx + dx
    map_y = yy + dy
    nx = 2.0*map_x/max(w-1,1) - 1.0
    ny = 2.0*map_y/max(h-1,1) - 1.0
    grid = torch.stack([nx,ny], dim=-1)
    warped = F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return warped, full