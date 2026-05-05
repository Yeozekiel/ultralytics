"""
MANGA-YOLO Training Script
Mamba-Inspired YOLO with Group Attention for Breast Mass Detection

Paper: MANGA-YOLO: A Mamba-inspired YOLO model with group attention for
       breast mass detection in mammograms
       Computers in Biology and Medicine 199 (2025) 111339

Architecture:
  - MACA (Mamba-Inspired Attention with Contextual Awareness) blocks in backbone
  - SCGA (Spatial and Channel Group Attention) blocks before each detection head
  - CIoU bounding box regression loss (already default in Ultralytics)
  - Input: 640x640

Usage:
  python train_manga_yolo.py --data dataset_vindr.yaml --epochs 300
"""

import argparse
import multiprocessing
import sys
from pathlib import Path

# ── make the repo root importable ────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Train MANGA-YOLO")
    parser.add_argument("--data", type=str, default="dataset_vindr.yaml",
                        help="Dataset YAML (dataset_vindr.yaml / dataset_cmmd.yaml / dataset_cbis_ddsm.yaml)")
    parser.add_argument("--model", type=str,
                        default="ultralytics/cfg/models/11/manga-yolo.yaml",
                        help="Model YAML config path")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Number of training epochs (paper: 300)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (paper: 640)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (paper: 32, reduce if OOM)")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers (0 = main thread, safest on Windows)")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device id or 'cpu'")
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="Initial learning rate (paper: 0.01 with AdamW)")
    parser.add_argument("--optimizer", type=str, default="AdamW",
                        help="Optimizer (paper: AdamW)")
    parser.add_argument("--scale", type=str, default="n",
                        choices=["n", "s", "m", "l", "x"],
                        help="Model scale (n=nano, s=small, m=medium, l=large, x=xlarge)")
    parser.add_argument("--project", type=str, default="runs/manga_yolo",
                        help="Output project directory")
    parser.add_argument("--name", type=str, default="train",
                        help="Experiment name")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    from ultralytics import YOLO

    print("=" * 60)
    print("MANGA-YOLO Training")
    print("=" * 60)
    print(f"  Model config : {args.model}")
    print(f"  Model scale  : {args.scale}")
    print(f"  Dataset      : {args.data}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Image size   : {args.imgsz}")
    print(f"  Batch size   : {args.batch}")
    print(f"  Optimizer    : {args.optimizer} (lr0={args.lr0})")
    print(f"  Device       : {args.device}")
    print("=" * 60)

    # Build model from YAML
    model = YOLO(args.model)
    model.info()

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        optimizer=args.optimizer,
        lr0=args.lr0,
        # Paper-aligned settings
        box=7.5,          # box loss weight (ultralytics default)
        cls=0.5,          # cls loss weight
        dfl=1.5,          # DFL loss weight
        # Augmentation (mild, appropriate for medical imaging)
        hsv_h=0.0,        # no hue shift (grayscale-like mammograms)
        hsv_s=0.0,        # no saturation shift
        hsv_v=0.4,        # small brightness variation
        degrees=0.0,      # no rotation
        translate=0.1,
        scale=0.5,
        fliplr=0.5,       # horizontal flip augmentation
        flipud=0.0,
        mosaic=0.0,       # disable mosaic (medical context)
        mixup=0.0,
        # Output
        project=args.project,
        name=args.name,
        resume=args.resume,
        plots=True,
        save=True,
        save_period=50,   # save checkpoint every 50 epochs
        val=True,
    )

    print("\n✅ Training complete!")
    print(f"   Results saved to: {results.save_dir}")
    print(f"   Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")

    return results


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
