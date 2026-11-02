import os
import glob
import torch

ckpt_dir = r"experiments\classifier_real_v2\checkpoints"
for p in sorted(glob.glob(os.path.join(ckpt_dir, "*.pth"))):
    data = torch.load(p, map_location="cpu")
    fname = os.path.basename(p)
    f1 = data.get("val_f1", "N/A")
    best_ep = data.get("best_epoch", "N/A")
    cfg = data.get("config", {})
    vm = data.get("val_metrics", {})
    acc = vm.get("accuracy", "N/A") if isinstance(vm, dict) else "N/A"
    print(f"{fname:<25} | Epoch: {best_ep} | Val F1: {f1} | Acc: {acc}")
