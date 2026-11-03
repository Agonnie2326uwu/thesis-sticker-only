"""
Train YOLO26n and YOLO11n for the PSU vehicle sticker benchmark.

Matches thesis Chapter IV, Step 6:
  - same data split for both models (70/15/15 = 980/210/210 images)
  - official COCO-pretrained weights (yolo26n.pt, yolo11n.pt)
  - imgsz 640, batch 16, 100 epochs, patience = epochs, optimizer = auto
  - three runs per model with seeds 0, 1, 2
  - brightness (hsv_v) and scale augmentation set to 0
  - best checkpoint chosen by validation mAP@0.5:0.95 (Ultralytics default)
  - the selected optimizer, learning rate, training time and best epoch
    are recorded for SOP 1

Usage (from the project folder, with the venv active):
    python train.py                 # train everything
    python train.py --check-only    # only check the split and settings
    python train.py --models YOLO26n --seeds 0

Outputs:
    runs/benchmark/<model>_s<seed>/weights/best.pt
    research_outputs/training_summary.csv
    research_outputs/training_runs.jsonl
"""

import argparse
import csv
import json
import platform
import time
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path

import yaml


# ============================================================
# CONFIGURATION — edit DATA_YAML to point at your dataset file
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Either an Ultralytics Platform dataset address (as used in your earlier
# training runs) or the path to a local data.yaml file.
DATA_YAML = "ul://carl-andre-de-castro/datasets/student-vehicle-sticker"

RUNS_DIR = BASE_DIR / "runs" / "benchmark"
OUTPUT_DIR = BASE_DIR / "research_outputs"

EXPECTED_SPLIT = {"train": 980, "val": 210, "test": 210}

PRETRAINED = {
    "YOLO26n": "yolo26n.pt",
    "YOLO11n": "yolo11n.pt",
}

SEEDS = [0, 1, 2]

# Identical settings for both models (thesis Chapter IV, Step 6).
TRAIN_SETTINGS = {
    "imgsz": 640,          # change to 1280 for BOTH models if Step 5 says so
    "batch": 16,
    "epochs": 100,
    "patience": 100,       # equal to epochs, so early stopping never triggers
    "optimizer": "auto",   # Ultralytics picks AdamW for runs under 10,000 iterations
    "hsv_v": 0.0,          # no brightness augmentation (keeps lighting out of training)
    "scale": 0.0,          # no scale augmentation (keeps distance out of training)
    "deterministic": True,
    "plots": True,
    "verbose": True,
}

# YOLO26n: validate and select the best checkpoint with the same NMS-free
# one-to-one head used at test time. In Ultralytics 8.4.142+ checkpoint
# selection follows the selected inference head.
MODEL_SETTINGS = {
    "YOLO26n": {"end2end": True},
    "YOLO11n": {},
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ============================================================
# DATASET CHECKS
# ============================================================

def _resolve(entry, root):
    path = Path(entry)
    return path if path.is_absolute() else (root / path)


def count_images(entry, root):
    """Count images for one split entry (folder, .txt list, or a list)."""
    if entry is None:
        return 0

    if isinstance(entry, (list, tuple)):
        return sum(count_images(item, root) for item in entry)

    path = _resolve(entry, root)

    if path.is_file() and path.suffix == ".txt":
        lines = path.read_text(encoding="utf-8").splitlines()
        return sum(1 for line in lines if line.strip())

    if path.is_dir():
        return sum(
            1 for item in path.rglob("*")
            if item.suffix.lower() in IMAGE_SUFFIXES
        )

    return 0


def load_dataset_config(data):
    """
    Return (config, root) for a local data.yaml or an Ultralytics Platform
    address (ul://...). Platform datasets are resolved, and downloaded if
    needed, by Ultralytics itself, the same way training does it.
    """
    local = Path(str(data))

    if local.is_file():
        config = yaml.safe_load(local.read_text(encoding="utf-8")) or {}
        root = _resolve(config.get("path", local.parent), local.parent)
        return config, root

    from ultralytics.data.utils import check_det_dataset

    config = check_det_dataset(str(data))
    root = Path(config.get("path", "."))
    return config, root


SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val",
                 "validation": "val", "test": "test"}


def find_ndjson(data):
    """Find the .ndjson file Ultralytics downloaded for a Platform dataset."""
    slug = str(data).rstrip("/").split("/")[-1]
    candidates = []

    for folder in (Path.cwd(), BASE_DIR):
        candidates += list(folder.rglob(f"{slug}.ndjson"))

    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def count_ndjson(path):
    """
    Count images per split in an Ultralytics Platform .ndjson export.
    Returns (counts, class_names, unknown_splits, first_record_keys).
    """
    counts = {"train": 0, "val": 0, "test": 0}
    unknown = {}
    names = {}
    first_keys = []

    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            if record.get("type") == "dataset":
                raw = record.get("class_names") or record.get("names") or {}
                names = (dict(enumerate(raw)) if isinstance(raw, list)
                         else {int(k): v for k, v in raw.items()})
                continue

            if not first_keys:
                first_keys = sorted(record)

            split = SPLIT_ALIASES.get(str(record.get("split", "")).lower())

            if split:
                counts[split] += 1
            else:
                key = str(record.get("split"))
                unknown[key] = unknown.get(key, 0) + 1

    return counts, names, unknown, first_keys


def check_dataset(data_yaml):
    """Return (counts, problems). Problems is a list of messages."""
    problems = []

    if str(data_yaml).startswith("ul://") or str(data_yaml).endswith(".ndjson"):
        path = (Path(data_yaml) if str(data_yaml).endswith(".ndjson")
                else find_ndjson(data_yaml))

        if path is None:
            try:   # let Ultralytics download it, then look again
                load_dataset_config(data_yaml)
            except Exception:
                pass
            path = find_ndjson(data_yaml)

        if path is None or not path.is_file():
            return {}, [f"Could not find the downloaded .ndjson file for "
                        f"{data_yaml}. Run 'yolo login' and try again."]

        print(f"Reading: {path}")
        counts, names, unknown, first_keys = count_ndjson(path)

        if unknown:
            problems.append(f"Images with no train/val/test split: {unknown}. "
                            f"Record fields: {first_keys}")

        for split, expected in EXPECTED_SPLIT.items():
            if counts[split] != expected:
                problems.append(f"{split}: found {counts[split]} images, "
                                f"thesis says {expected}.")

        if len(names) != 1:
            problems.append(
                f"Dataset has {len(names)} classes {names}; the thesis "
                "benchmark uses one class (PSU Student Vehicle Sticker)."
            )

        return counts, problems

    try:
        config, root = load_dataset_config(data_yaml)
    except Exception as exc:
        return {}, [
            f"Could not load the dataset '{data_yaml}': {exc}. "
            "If it is a Platform address, check that you are logged in "
            "to the Ultralytics Platform (yolo login)."
        ]

    counts = {
        split: count_images(config.get(split), root)
        for split in ("train", "val", "test")
    }

    for split, expected in EXPECTED_SPLIT.items():
        if config.get(split) is None:
            problems.append(f"'{split}' is missing from {data_yaml.name}.")
        elif counts[split] != expected:
            problems.append(
                f"{split}: found {counts[split]} images, "
                f"thesis says {expected}."
            )

    names = config.get("names", {})
    if isinstance(names, list):
        names = dict(enumerate(names))

    if len(names) != 1:
        problems.append(
            f"Dataset has {len(names)} classes {names}; the thesis "
            "benchmark uses one class (PSU Student Vehicle Sticker). "
            "Remove the privacy-blur class from the benchmark dataset."
        )

    return counts, problems


def check_environment():
    """Warnings about things that would break the thesis settings."""
    warnings = []

    if find_spec("albumentations") is not None:
        warnings.append(
            "The albumentations package is installed. Ultralytics would "
            "add random blur during training. Uninstall it first: "
            "pip uninstall albumentations"
        )

    return warnings


# ============================================================
# TRAINING
# ============================================================

def best_epoch_from_results(results_csv):
    """Read Ultralytics' results.csv and return the best epoch row."""
    if not results_csv.is_file():
        return {}

    with results_csv.open(newline="", encoding="utf-8") as file:
        rows = [
            {key.strip(): value for key, value in row.items()}
            for row in csv.DictReader(file)
        ]

    key = "metrics/mAP50-95(B)"
    rows = [row for row in rows if row.get(key)]

    if not rows:
        return {}

    best = max(rows, key=lambda row: float(row[key]))

    return {
        "best_epoch": int(float(best["epoch"])),
        "best_val_map50_95": float(best[key]),
        "best_val_map50": float(best.get("metrics/mAP50(B)", "nan")),
        "epochs_completed": len(rows),
    }


def train_one(model_name, seed):
    from ultralytics import YOLO
    import ultralytics
    import torch

    run_name = f"{model_name.lower()}_s{seed}"
    model = YOLO(PRETRAINED[model_name])

    settings = {
        **TRAIN_SETTINGS,
        **MODEL_SETTINGS.get(model_name, {}),
        "data": str(DATA_YAML),
        "seed": seed,
        "project": str(RUNS_DIR),
        "name": run_name,
        "exist_ok": False,
        "device": 0 if torch.cuda.is_available() else "cpu",
    }

    started = time.perf_counter()
    model.train(**settings)
    train_seconds = time.perf_counter() - started

    trainer = model.trainer
    save_dir = Path(trainer.save_dir)
    optimizer = getattr(trainer, "optimizer", None)

    record = {
        "model": model_name,
        "seed": seed,
        "run_dir": str(save_dir),
        "best_checkpoint": str(save_dir / "weights" / "best.pt"),
        "train_time_s": round(train_seconds, 1),
        "selected_optimizer": type(optimizer).__name__ if optimizer else "",
        "initial_lr": (
            optimizer.param_groups[0].get("initial_lr",
                                          optimizer.param_groups[0]["lr"])
            if optimizer else ""
        ),
        "amp": bool(getattr(trainer, "amp", False)),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else platform.processor() or "CPU"
        ),
        "settings": {k: v for k, v in settings.items() if k != "device"},
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        **best_epoch_from_results(save_dir / "results.csv"),
    }

    return record


def save_record(record):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUTPUT_DIR / "training_runs.jsonl").open(
        "a", encoding="utf-8"
    ) as file:
        file.write(json.dumps(record) + "\n")

    summary = OUTPUT_DIR / "training_summary.csv"
    columns = [
        "model", "seed", "best_epoch", "epochs_completed",
        "best_val_map50_95", "best_val_map50", "train_time_s",
        "selected_optimizer", "initial_lr", "amp", "ultralytics_version",
        "torch_version", "python_version", "device", "best_checkpoint",
    ]
    new_file = not summary.exists()

    with summary.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns,
                                extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(record)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--models", nargs="+", default=list(PRETRAINED),
                        choices=list(PRETRAINED))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--check-only", action="store_true",
                        help="check the dataset split and stop")
    parser.add_argument("--allow-split-mismatch", action="store_true",
                        help="train even if the split is not 980/210/210")
    args = parser.parse_args()

    counts, problems = check_dataset(DATA_YAML)
    warnings = check_environment()

    print(f"Dataset: {DATA_YAML}")
    print(f"Images found: {counts}  (thesis: {EXPECTED_SPLIT})")

    for message in problems:
        print(f"  PROBLEM: {message}")
    for message in warnings:
        print(f"  WARNING: {message}")

    if args.check_only:
        print("Check finished." if not problems else "Fix the problems above.")
        return

    if problems and not args.allow_split_mismatch:
        raise SystemExit(
            "Stopping: the dataset does not match the thesis. Fix it, or "
            "rerun with --allow-split-mismatch and update Chapter IV."
        )

    for model_name in args.models:
        for seed in args.seeds:
            print(f"\n=== Training {model_name}, seed {seed} ===")
            record = train_one(model_name, seed)
            save_record(record)
            print(
                f"Done: best mAP50-95 {record.get('best_val_map50_95')} "
                f"at epoch {record.get('best_epoch')}, "
                f"optimizer {record['selected_optimizer']}, "
                f"{record['train_time_s']} s"
            )


if __name__ == "__main__":
    main()