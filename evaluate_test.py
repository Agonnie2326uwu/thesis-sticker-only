"""
Evaluate the final YOLO26n and YOLO11n checkpoints on the held-out test set.

This produces the main result table for the thesis (SOP 2 and SOP 3):
  1. Accuracy on the 210 test images, from Ultralytics' validator:
       - precision, recall, F1
       - mAP@0.5 and mAP@0.5:0.95 (COCO-style, 101-point interpolation)
       - TP / FP / FN counts at the app's confidence threshold (0.25)
       - mean confidence of the stickers each model finds, and mean confidence
         over all labelled stickers (a missed sticker counts as 0)
  2. Speed on the same 210 test images, one image at a time (batch 1):
       - one full untimed warm-up pass per model first
       - three timed runs per model, swapping the model order each run
       - FPS as mean and SD of the three runs, end to end and model only
       - preprocess / inference / postprocess time per image
       - mean, standard deviation, median and 95th percentile
       - FPS, on the GPU and on the CPU

Both models are evaluated the same way. YOLO26n uses its NMS-free one-to-one
head (nms=False), exactly as in app.py. YOLO11n uses standard NMS.

The script checks that it runs on the RTX 2050 laptop, the machine named in
the thesis (see EXPECTED_GPU).

Usage (from the project folder, with the venv active):
    python evaluate_test.py                  # accuracy + GPU and CPU speed
    python evaluate_test.py --skip-speed     # accuracy only
    python evaluate_test.py --devices gpu    # speed on the GPU only

Outputs:
    research_outputs/test_metrics.csv       accuracy table (one row per model)
    research_outputs/test_speed.csv         speed table (one row per model and device)
    research_outputs/test_evaluation.jsonl  everything, with versions and settings
    runs/test_eval/<model>/                  PR curve, F1 curve, confusion matrix
"""

import argparse
import csv
import hashlib
import json
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# The dataset on the Ultralytics Platform (the address train.py uses).
PLATFORM_DATA = "ul://carl-andre-de-castro/datasets/student-vehicle-sticker"

# The exact copy of the dataset that the final models were trained and tested
# on (downloaded by Ultralytics on this computer before some photos were
# re-uploaded to the Platform in Nov 2026). The tests use this copy, so every
# result refers to the same 210 test photos. If it is not on this computer
# (e.g. the classmate's laptop), the Platform address is used instead.
ORIGINAL_DATASET = Path.home() / "datasets" / "student-vehicle-sticker-b0394f94"
ORIGINAL_YAML = BASE_DIR / "original_dataset.yaml"


def default_data():
    """Local data.yaml of the original dataset copy, else the Platform."""
    images = ORIGINAL_DATASET / "images"
    if not (images / "test").is_dir():
        return PLATFORM_DATA
    for found in sorted(ORIGINAL_DATASET.glob("*.yaml")):
        return str(found)
    # No yaml in the copy: write one. Ultralytics requires train and val
    # entries; the tests only read the test split.
    test = "images/test"
    train = "images/train" if (images / "train").is_dir() else test
    val = "images/val" if (images / "val").is_dir() else test
    ORIGINAL_YAML.write_text(
        f"path: {ORIGINAL_DATASET.as_posix()}\ntrain: {train}\nval: {val}\n"
        f"test: {test}\nnames:\n  0: student-vehicle-sticker\n",
        encoding="utf-8")
    return str(ORIGINAL_YAML)


DATA = default_data()

FINAL_RUNS_DIR = BASE_DIR / "runs" / "detect" / "carl-andre-de-castro" / "runs"

MODEL_PATHS = {
    "YOLO26n": FINAL_RUNS_DIR / "yolo26n_final_70_15_15" / "weights" / "best.pt",
    "YOLO11n": FINAL_RUNS_DIR / "yolo11n_final_70_15_15-2" / "weights" / "best.pt",
}

# Inference head for each model, identical to app.py.
HEADS = {
    "YOLO26n": {"label": "one-to-one (NMS-free)", "args": {"nms": False}},
    "YOLO11n": {"label": "NMS", "args": {}},
}

EXPECTED_TEST_IMAGES = 210
STICKER_KEYWORD = "sticker"

IMAGE_SIZE = 640
BATCH = 16                 # for the accuracy run only; speed uses batch 1
MAP_CONF = 0.001           # Ultralytics' standard confidence floor for mAP
APP_CONF = 0.25            # the app's default threshold, for TP / FP / FN
NMS_IOU = 0.7              # Ultralytics default NMS IoU (YOLO11n only)
MATCH_IOU = 0.5            # a detection matches a sticker at IoU >= 0.5
                           # (same rule as the field test)

OUTPUT_DIR = BASE_DIR / "research_outputs"
EVAL_RUNS_DIR = BASE_DIR / "runs" / "test_eval"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ============================================================
# SMALL HELPERS
# ============================================================

def timestamp_now():
    return datetime.now().isoformat(timespec="seconds")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def f1_score(precision, recall):
    total = precision + recall
    return 2 * precision * recall / total if total > 0 else 0.0


def safe_div(a, b):
    return a / b if b else 0.0


def percentile(values, q):
    """Linear-interpolation percentile (same method as numpy's default)."""
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(values):
    """Mean, SD, median and 95th percentile of a list of milliseconds."""
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "p95": percentile(values, 95),
    }


def sticker_class_ids(names):
    """Class ids whose name contains the sticker keyword."""
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    return [int(i) for i, n in names.items() if STICKER_KEYWORD in str(n).lower()]


def round_dict(record, digits=5):
    return {
        key: round(value, digits) if isinstance(value, float) else value
        for key, value in record.items()
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = []
    for row in rows:
        columns += [key for key in row if key not in columns]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, default=str) + "\n")


def environment_info():
    import torch
    import ultralytics

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "gpu": gpu,
        "cpu_threads": torch.get_num_threads(),
    }


# Thesis Environment / Limitations section: all three tests (the natural test
# set, the FPS benchmark, the synthetic stressors and the field test) are
# measured on the RTX 2050 laptop. Each script stops if it finds a different
# GPU, so results from the wrong machine cannot end up in the tables by
# mistake.
EXPECTED_GPU = "RTX 2050"


def check_test_gpu(allow_other, expected=EXPECTED_GPU):
    """Stop unless the GPU is the one the thesis names (or the user overrides)."""
    import torch

    found = (torch.cuda.get_device_name(0) if torch.cuda.is_available()
             else "no CUDA GPU")

    if expected.lower() in found.lower():
        return found

    message = (f"This computer's GPU is '{found}', but the thesis says this "
               f"test runs on the {expected}.")
    if not allow_other:
        raise SystemExit(
            message + f"\nRun this script on the {expected} computer, or add "
            "--allow-other-gpu if you really mean to use this computer "
            "(then update the thesis text to match).")
    print("WARNING: " + message + " Continuing because of --allow-other-gpu.")
    return found


# ============================================================
# DATASET
# ============================================================

def resolve_test_images(data):
    """
    Return (data_yaml_path, list_of_test_image_paths, class_names).
    Platform datasets are converted to YOLO format by Ultralytics itself,
    the same conversion its validator uses.
    """
    from ultralytics.data.utils import (check_det_dataset,
                                        convert_ndjson_to_yolo_if_needed)

    data_yaml = convert_ndjson_to_yolo_if_needed(str(data), 1.0, split="test")
    config = check_det_dataset(str(data_yaml))

    if not config.get("test"):
        raise SystemExit(
            "The dataset has no 'test' split. The thesis needs the 210 "
            "held-out test images."
        )

    images = collect_images(config["test"], Path(config.get("path", ".")))
    return data_yaml, images, config.get("names", {})


def collect_images(entry, root):
    if isinstance(entry, (list, tuple)):
        found = []
        for item in entry:
            found += collect_images(item, root)
        return sorted(set(found))

    path = Path(entry)
    path = path if path.is_absolute() else root / path

    if path.is_file() and path.suffix == ".txt":
        lines = path.read_text(encoding="utf-8").splitlines()
        return sorted(
            (Path(line.strip()) if Path(line.strip()).is_absolute()
             else root / line.strip())
            for line in lines if line.strip()
        )

    if path.is_dir():
        return sorted(
            p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
        )

    return []


# ============================================================
# ACCURACY
# ============================================================

def sticker_metrics(box, class_id):
    """
    Precision, recall, mAP@0.5 and mAP@0.5:0.95 for the sticker class.
    With a single-class dataset this equals the overall result.
    """
    ap_classes = [int(c) for c in list(box.ap_class_index)]

    if class_id not in ap_classes:
        return {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map50_95": 0.0}

    p, r, ap50, ap = box.class_result(ap_classes.index(class_id))
    return {"precision": float(p), "recall": float(r),
            "map50": float(ap50), "map50_95": float(ap)}


def confusion_counts(matrix, class_id):
    """TP, FP, FN for one class from Ultralytics' (nc+1) x (nc+1) matrix."""
    tp = float(matrix[class_id][class_id])
    fp = float(sum(matrix[class_id])) - tp
    fn = float(sum(row[class_id] for row in matrix)) - tp
    return int(round(tp)), int(round(fp)), int(round(fn))


def box_iou(a, b):
    """Intersection over union of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
             - inter)
    return inter / union if union > 0 else 0.0


def label_file(image_path):
    """The YOLO label file that belongs to an image (images/ -> labels/)."""
    parts = list(Path(image_path).parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def read_ground_truth(image_path, class_id):
    """Sticker boxes (xyxy, pixels) from the image's YOLO label file."""
    import cv2

    labels = label_file(image_path)
    if not labels.is_file():
        raise SystemExit(
            f"No label file found for test image:\n  {image_path}\n"
            f"Expected:\n  {labels}\n"
            "The confidence score needs the correct sticker boxes.")

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read test image: {image_path}")
    height, width = image.shape[:2]

    boxes = []
    for line in labels.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5 or int(float(fields[0])) != class_id:
            continue
        cx, cy, bw, bh = (float(value) for value in fields[1:5])
        boxes.append(((cx - bw / 2) * width, (cy - bh / 2) * height,
                      (cx + bw / 2) * width, (cy + bh / 2) * height))
    return boxes


def match_confidences(truth, predictions):
    """
    Greedy one-to-one matching, highest confidence first (the same rule as
    the field test). Returns the confidence of the detection matched to each
    truth box; stickers with no match are left out.
    """
    matched = []
    taken = set()

    for box, conf in sorted(predictions, key=lambda p: -p[1]):
        best, best_iou = None, MATCH_IOU
        for index, gt in enumerate(truth):
            if index in taken:
                continue
            value = box_iou(box, gt)
            if value >= best_iou:
                best, best_iou = index, value
        if best is not None:
            taken.add(best)
            matched.append(conf)

    return matched


def confidence_scores(model, head, images, class_id, device):
    """
    Mean detection confidence on the 210 test images, per labelled sticker.

    mean_conf_detected      mean confidence over the stickers that were found
    mean_conf_all_stickers  mean over every labelled sticker, with a missed
                            sticker counting as 0 (so recall is included)
    """
    found, every = [], []

    for path in images:
        truth = read_ground_truth(path, class_id)
        result = model.predict(
            source=str(path), conf=APP_CONF, iou=NMS_IOU, imgsz=IMAGE_SIZE,
            device=device, verbose=False, **head["args"])[0]

        predictions = []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            predictions = list(zip(boxes.xyxy.cpu().numpy().tolist(),
                                   boxes.conf.cpu().numpy().tolist()))

        matched = match_confidences(truth, predictions)
        found.extend(matched)
        every.extend(matched)
        every.extend([0.0] * (len(truth) - len(matched)))

    return {
        "matched_stickers_at_app_conf": len(found),
        "mean_conf_detected": statistics.fmean(found) if found else 0.0,
        "sd_conf_detected": statistics.stdev(found) if len(found) > 1 else 0.0,
        "mean_conf_all_stickers": statistics.fmean(every) if every else 0.0,
        "sd_conf_all_stickers": (statistics.stdev(every)
                                 if len(every) > 1 else 0.0),
    }


def evaluate_accuracy(name, checkpoint, head, data_yaml, device, images):
    from ultralytics import YOLO

    common = {
        "data": str(data_yaml),
        "split": "test",
        "imgsz": IMAGE_SIZE,
        "batch": BATCH,
        "iou": NMS_IOU,
        "device": device,
        "project": str(EVAL_RUNS_DIR),
        "exist_ok": True,
        "verbose": False,
        **head["args"],
    }
    folder = name.replace(" ", "_").replace("(", "").replace(")", "")

    # Run 1: standard mAP evaluation (confidence floor 0.001) with plots.
    model = YOLO(str(checkpoint))
    class_ids = sticker_class_ids(model.names)

    if not class_ids:
        raise SystemExit(f"{name}: no class name contains '{STICKER_KEYWORD}'. "
                         f"Classes: {model.names}")

    class_id = class_ids[0]
    metrics = model.val(conf=MAP_CONF, plots=True, name=folder, **common)
    scores = sticker_metrics(metrics.box, class_id)

    # Run 2: the app's threshold (0.25), for TP / FP / FN counts.
    # plots=True is required: Ultralytics only fills the confusion matrix
    # when plotting is on. Its plots go to a separate "_conf0.25" folder.
    model = YOLO(str(checkpoint))
    at_app = model.val(conf=APP_CONF, plots=True,
                       name=f"{folder}_conf{APP_CONF}", **common)
    tp, fp, fn = confusion_counts(at_app.confusion_matrix.matrix.tolist(),
                                  class_id)
    instances = int(getattr(at_app, "nt_per_class", [0])[class_id]) \
        if getattr(at_app, "nt_per_class", None) is not None else None

    if tp + fn == 0:
        raise SystemExit(
            f"{name}: the confusion matrix is empty (TP = FN = 0), so the "
            "counts at the app threshold cannot be trusted. Please report "
            "this output."
        )

    # Run 3: confidence of each matched sticker at the app's threshold.
    conf = confidence_scores(model, head, images, class_id, device)

    return round_dict({
        "model": name,
        "inference_head": head["label"],
        "class_name": model.names[class_id],
        "num_classes_in_model": len(model.names),
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f1": f1_score(scores["precision"], scores["recall"]),
        "map50": scores["map50"],
        "map50_95": scores["map50_95"],
        "app_conf": APP_CONF,
        "labelled_stickers": instances if instances else tp + fn,
        "tp_at_app_conf": tp,
        "fp_at_app_conf": fp,
        "fn_at_app_conf": fn,
        "precision_at_app_conf": safe_div(tp, tp + fp),
        "recall_at_app_conf": safe_div(tp, tp + fn),
        "f1_at_app_conf": f1_score(safe_div(tp, tp + fp), safe_div(tp, tp + fn)),
        "matched_stickers_at_app_conf": conf["matched_stickers_at_app_conf"],
        "mean_conf_detected": conf["mean_conf_detected"],
        "sd_conf_detected": conf["sd_conf_detected"],
        "mean_conf_all_stickers": conf["mean_conf_all_stickers"],
        "sd_conf_all_stickers": conf["sd_conf_all_stickers"],
        "val_ms_per_image": sum(
            metrics.speed.get(k, 0.0) for k in ("preprocess", "inference",
                                                "postprocess")
        ),
        "plots_folder": str(Path(metrics.save_dir)),
    })


# ============================================================
# SPEED
# ============================================================

def load_images(paths):
    import cv2

    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"Could not read test image: {path}")
        images.append(image)
    return images


SPEED_ROUNDS = 3           # thesis SOP 5: three timed runs per model and device;
                           # the model order is swapped every run


def benchmark_device(jobs, images, device, device_label):
    """
    Time one-image-at-a-time inference for every model on one device.

    Fairness rules (the first model timed must not pay start-up costs):
      1. Every model first does a full untimed warm-up pass over all test
         images, so GPU kernels are compiled for every image shape.
      2. Then SPEED_ROUNDS timed passes, swapping the model order each
         round (A, B, then B, A), so neither model always runs first.
    Returns one summary row per model.
    """
    from ultralytics import YOLO

    models = {name: (YOLO(str(checkpoint)), head)
              for name, checkpoint, head in jobs}

    def run(name, image):
        model, head = models[name]
        return model.predict(
            source=image, conf=APP_CONF, iou=NMS_IOU, imgsz=IMAGE_SIZE,
            device=device, verbose=False, **head["args"],
        )[0]

    for name in models:                       # untimed warm-up
        print(f"  warm-up: {name} ({len(images)} images)")
        for image in images:
            run(name, image)

    samples = {name: {"pre": [], "inf": [], "post": [], "total": [],
                      "wall": [], "round_means": [],
                      "round_model_means": []} for name in models}
    order = list(models)

    for round_no in range(SPEED_ROUNDS):
        for name in order:
            print(f"  timed round {round_no + 1}/{SPEED_ROUNDS}: {name}")
            data = samples[name]
            round_totals, round_models = [], []

            for image in images:
                start = time.perf_counter()
                result = run(name, image)
                data["wall"].append((time.perf_counter() - start) * 1000)

                speed = result.speed
                pre = speed.get("preprocess", 0.0)
                inf = speed.get("inference", 0.0)
                post = speed.get("postprocess", 0.0)
                data["pre"].append(pre)
                data["inf"].append(inf)
                data["post"].append(post)
                data["total"].append(pre + inf + post)
                round_totals.append(pre + inf + post)
                round_models.append(inf)

            data["round_means"].append(statistics.fmean(round_totals))
            data["round_model_means"].append(statistics.fmean(round_models))
        order.reverse()

    rows = []
    for name, (_model, head) in models.items():
        data = samples[name]
        total_stats = summarize(data["total"])
        wall_stats = summarize(data["wall"])

        row = {
            "model": name,
            "inference_head": head["label"],
            "device": device_label,
            "images_per_round": len(images),
            "timed_rounds": SPEED_ROUNDS,
            "timed_samples": len(data["total"]),
            "warmup": f"1 full untimed pass ({len(images)} images)",
            "batch": 1,
            "imgsz": IMAGE_SIZE,
        }
        for label, key in (("preprocess", "pre"), ("inference", "inf"),
                           ("postprocess", "post")):
            stats = summarize(data[key])
            row[f"{label}_ms_mean"] = stats["mean"]
            row[f"{label}_ms_sd"] = stats["sd"]

        row.update({
            "total_ms_mean": total_stats["mean"],
            "total_ms_sd": total_stats["sd"],
            "total_ms_median": total_stats["median"],
            "total_ms_p95": total_stats["p95"],
            "fps": safe_div(1000, total_stats["mean"]),
            "wall_ms_mean": wall_stats["mean"],
            "wall_fps": safe_div(1000, wall_stats["mean"]),
            "round_total_ms_means": " / ".join(
                f"{m:.3f}" for m in data["round_means"]),
        })

        # Thesis SOP 5: FPS as the mean and SD of the runs (one FPS value
        # per run), end to end and model only.
        run_fps = [safe_div(1000, m) for m in data["round_means"]]
        run_model_fps = [safe_div(1000, m)
                         for m in data["round_model_means"]]
        row.update({
            "fps_runs_mean": statistics.fmean(run_fps),
            "fps_runs_sd": (statistics.stdev(run_fps)
                            if len(run_fps) > 1 else 0.0),
            "model_only_fps_runs_mean": statistics.fmean(run_model_fps),
            "model_only_fps_runs_sd": (statistics.stdev(run_model_fps)
                                       if len(run_model_fps) > 1 else 0.0),
        })
        rows.append(round_dict(row, 3))
    return rows


# ============================================================
# MAIN
# ============================================================

def print_table(title, rows, columns):
    print(f"\n{title}")
    widths = [max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(w) for c, w in zip(columns, widths)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data", default=DATA,
                        help="Platform address or local data.yaml")
    parser.add_argument("--models", nargs="+", default=list(MODEL_PATHS))
    parser.add_argument("--devices", nargs="+", default=["gpu", "cpu"],
                        choices=["gpu", "cpu"])
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--allow-count-mismatch", action="store_true")
    parser.add_argument("--allow-other-gpu", action="store_true",
                        help=f"run even if the GPU is not the "
                             f"{EXPECTED_GPU}")
    args = parser.parse_args(argv)

    import torch

    # ---- models to evaluate -----------------------------------------
    jobs = []
    for name in args.models:
        if name not in MODEL_PATHS:
            raise SystemExit(f"Unknown model {name}. Choose from {list(MODEL_PATHS)}")
        jobs.append((name, MODEL_PATHS[name], HEADS[name]))

    for name, checkpoint, _ in jobs:
        if not checkpoint.is_file():
            raise SystemExit(f"{name}: checkpoint not found:\n  {checkpoint}")

    # ---- dataset ----------------------------------------------------
    print(f"Dataset: {args.data}")
    data_yaml, test_images, names = resolve_test_images(args.data)
    print(f"Test images found: {len(test_images)}")

    if len(test_images) != EXPECTED_TEST_IMAGES and not args.allow_count_mismatch:
        raise SystemExit(
            f"Expected {EXPECTED_TEST_IMAGES} test images (thesis), found "
            f"{len(test_images)}. Check the dataset, or rerun with "
            "--allow-count-mismatch if the thesis number changed."
        )

    tested_gpu = check_test_gpu(args.allow_other_gpu)
    gpu_available = torch.cuda.is_available()
    accuracy_device = 0 if gpu_available else "cpu"
    env = environment_info()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    jsonl = OUTPUT_DIR / "test_evaluation.jsonl"

    checksums = {}
    for name, checkpoint, _ in jobs:
        checksums[name] = sha256_file(checkpoint)

    append_jsonl(jsonl, {
        "record_type": "evaluation_started", "run_id": run_id,
        "timestamp": timestamp_now(), "data": str(args.data),
        "data_yaml": str(data_yaml), "test_images": len(test_images),
        "dataset_names": names, "environment": env,
        "settings": {"imgsz": IMAGE_SIZE, "batch": BATCH, "map_conf": MAP_CONF,
                     "app_conf": APP_CONF, "nms_iou": NMS_IOU,
                     "speed_warmup": "1 full untimed pass per model",
                     "speed_rounds": SPEED_ROUNDS,
                     "expected_gpu": EXPECTED_GPU,
                     "tested_gpu": tested_gpu},
    })

    # ---- accuracy ---------------------------------------------------
    accuracy_rows = []
    for name, checkpoint, head in jobs:
        print(f"\n=== Accuracy: {name} [{head['label']}] ===")
        row = evaluate_accuracy(name, checkpoint, head, data_yaml,
                                accuracy_device, test_images)
        row.update({"test_images": len(test_images),
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": checksums[name],
                    "run_id": run_id})
        accuracy_rows.append(row)
        append_jsonl(jsonl, {"record_type": "accuracy", **row})

    write_csv(OUTPUT_DIR / "test_metrics.csv", accuracy_rows)

    # ---- speed ------------------------------------------------------
    speed_rows = []
    if not args.skip_speed:
        devices = []
        for label in args.devices:
            if label == "gpu" and not gpu_available:
                print("No CUDA GPU found — skipping GPU speed.")
                continue
            devices.append((0, f"GPU ({env['gpu']})") if label == "gpu"
                           else ("cpu", f"CPU ({env['cpu']}, "
                                        f"{env['cpu_threads']} threads)"))

        print(f"\nLoading {len(test_images)} test images into memory...")
        images = load_images(test_images)

        for device, device_label in devices:
            print(f"=== Speed on {device_label} ===")
            for row in benchmark_device(jobs, images, device, device_label):
                row["run_id"] = run_id
                speed_rows.append(row)
                append_jsonl(jsonl, {"record_type": "speed", **row})

        write_csv(OUTPUT_DIR / "test_speed.csv", speed_rows)

    # ---- summary ----------------------------------------------------
    print_table(
        f"TEST SET ACCURACY ({len(test_images)} images)",
        accuracy_rows,
        ["model", "precision", "recall", "f1", "map50", "map50_95",
         "labelled_stickers", "tp_at_app_conf", "fp_at_app_conf", "fn_at_app_conf",
         "mean_conf_detected", "mean_conf_all_stickers"],
    )
    if speed_rows:
        print_table(
            "SPEED (batch 1, after warm-up; ms per image)",
            speed_rows,
            ["model", "device", "preprocess_ms_mean", "inference_ms_mean",
             "postprocess_ms_mean", "total_ms_mean", "fps_runs_mean",
             "fps_runs_sd", "model_only_fps_runs_mean",
             "round_total_ms_means"],
        )

    print(f"\nSaved: {OUTPUT_DIR / 'test_metrics.csv'}")
    if speed_rows:
        print(f"Saved: {OUTPUT_DIR / 'test_speed.csv'}")
    print(f"Saved: {jsonl}")
    print(f"Plots: {EVAL_RUNS_DIR}")
    return accuracy_rows, speed_rows


if __name__ == "__main__":
    main()