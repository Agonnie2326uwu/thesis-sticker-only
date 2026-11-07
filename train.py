"""
Train YOLO26n and YOLO11n for the PSU vehicle sticker benchmark.

These are the exact settings of the two final models used in the thesis
(runs/detect/carl-andre-de-castro/runs/yolo26n_final_70_15_15 and
yolo11n_final_70_15_15-2, see their args.yaml). Both models are trained once,
with identical settings, on the same 980 / 210 / 210 split:

  - official COCO-pretrained weights (yolo26n.pt, yolo11n.pt)
  - input size 640, batch size 4, 100 epochs, patience 100 (all epochs run)
  - optimizer auto: Ultralytics counts iterations with its nominal batch of
    64 (gradients are accumulated over 16 batches of 4), so 980 images x
    100 epochs = about 1,600 iterations and it selects AdamW
  - Ultralytics default augmentation (hsv_h 0.015, hsv_s 0.7, hsv_v 0.4,
    translate 0.1, scale 0.5, fliplr 0.5, mosaic 1.0, off for the last 10
    epochs)
  - seed 0, deterministic mode
  - the best checkpoint (highest validation mAP@0.5:0.95) is saved as
    weights/best.pt

Note: after training, some photos on the Ultralytics Platform were replaced
with copies whose plate numbers are blurred. Training again from the
Platform would use those copies, so the new models would differ slightly
from the final ones.

The final models are already trained and included in this repository. Run
this only to reproduce them (on the GTX 1660 SUPER desktop):
    python train.py
    python train.py --models YOLO26n
"""

import argparse

# ============================================================
# SETTINGS
# ============================================================

DATA = "ul://carl-andre-de-castro/datasets/student-vehicle-sticker"

PRETRAINED = {
    "YOLO26n": "yolo26n.pt",
    "YOLO11n": "yolo11n.pt",
}

SETTINGS = {
    "data": DATA,
    "epochs": 100,
    "patience": 100,
    "batch": 4,
    "imgsz": 640,
    "optimizer": "auto",
    "seed": 0,
    "deterministic": True,
    "workers": 2,
    "project": "runs/detect/reproduced",
}


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train YOLO26n and YOLO11n")
    parser.add_argument("--models", nargs="+", default=list(PRETRAINED),
                        choices=list(PRETRAINED))
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"

    for name in args.models:
        print(f"\n=== Training {name} ===")
        model = YOLO(PRETRAINED[name])
        model.train(**SETTINGS, name=f"{name.lower()}_reproduced",
                    device=device)
        optimizer = type(model.trainer.optimizer).__name__
        print(f"{name} finished. Optimizer used: {optimizer}. "
              f"Best model: {model.trainer.best}")


if __name__ == "__main__":
    main()