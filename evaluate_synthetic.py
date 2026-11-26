"""
Test 3 (thesis Step 8, SOP 4 and SOP 5): synthetic stressor test.

For motion blur, occlusion, and lighting, copies of the baseline test images
are degraded one stressor at a time in the steps of Table 4.2 (synthetic
stressor sweeps), and both models are run on every copy.

  1. Baseline images: natural test images in which BOTH models detect every
     sticker (confidence >= 0.25, IoU >= 0.5) before any degradation.
  2. Sweeps (applied to the full-resolution image):
       motion_blur  horizontal linear motion kernel, L = 0 to 100 px, 5 px steps
       occlusion    dark band over the sticker, covering 0% to 90% of its box
                    (visible area 100% to 10%) in 10% steps
       low_light    gamma darkening  I' = 255 (I/255)^gamma, gamma 1.00-5.00
       glare        brightness gain  I' = clip(alpha I, 0, 255), alpha 1.00-3.00
     None of these moves the sticker. For occlusion the ground-truth box is
     reduced to the visible part, as in the Annotation Procedure (one box
     that tightly encloses the visible part of the sticker).
  3. Per step and model: detection rate (conf >= 0.25, IoU >= 0.5), mean
     confidence, precision and recall at 0.25, mAP@0.5 and mAP@0.5:0.95
     (Ultralytics' own AP function), and degradation (%) from step 0.
  4. Per stressor and model: logistic regression of the detection outcome on
     the step level gives x50 (breaking point) and x90 (reliable detection).
     The models are compared with 1,000 paired bootstrap resamples of the
     baseline images (x50 difference) and McNemar's exact test.
  5. SOP 5: the stressor reached with the least change from baseline, as the
     fraction of its tested range at which x50 falls, limits detection most.
     Distance is included automatically if evaluate_field.py has been run.

Run on the RTX 2050 laptop (the script checks):
    python evaluate_synthetic.py
    python evaluate_synthetic.py --max-images 5      # quick trial run
    python evaluate_synthetic.py --stressors occlusion glare

Outputs (research_outputs/synthetic_results/):
    baseline_images.csv          which test images are baseline images, and why
    synthetic_steps.csv          metrics per stressor, step and model
    synthetic_outcomes.csv       one row per sticker, step and stressor
    synthetic_breaking_points.csv  x50 / x90 per stressor and model
    synthetic_comparison.csv     bootstrap CI of the x50 difference, McNemar
    stressor_ranking.csv         SOP 5 ranking (with distance if available)
    chart_<stressor>.png         detection rate and mAP@0.5:0.95 per step
    examples/<stressor>.jpg      one baseline sticker at five steps (figures)
    synthetic_evaluation.jsonl   everything, with versions and settings
"""

import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import evaluate_field as ef
import evaluate_test as ev


# ============================================================
# CONFIGURATION (Table 4.2 in the thesis)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "research_outputs" / "synthetic_results"
FIELD_RESULTS = (BASE_DIR / "research_outputs" / "field_results"
                 / "field_breaking_points.csv")

SWEEPS = {
    "motion_blur": {
        "values": [5 * i for i in range(21)],                 # L, px
        "unit": "kernel length L (px)",
        "label": "Motion blur",
    },
    "occlusion": {
        "values": [round(0.1 * i, 1) for i in range(10)],     # share covered
        "unit": "share of sticker covered",
        "label": "Occlusion",
    },
    "low_light": {
        "values": [1 + 0.25 * i for i in range(17)],          # gamma
        "unit": "gamma",
        "label": "Low light (gamma darkening)",
    },
    "glare": {
        "values": [1 + 0.125 * i for i in range(17)],         # alpha
        "unit": "brightness gain alpha",
        "label": "Glare / overexposure",
    },
}

OCCLUSION_COLOUR = (25, 25, 25)   # dark band (BGR)
DETECT_CONF = ef.DETECT_CONF      # 0.25, thesis definition of "detected"
DETECT_IOU = ef.DETECT_IOU        # 0.50
MAP_CONF = ev.MAP_CONF            # 0.001, Ultralytics' floor for mAP
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
BOOTSTRAP_RESAMPLES = ef.BOOTSTRAP_RESAMPLES
BOOTSTRAP_SEED = ef.BOOTSTRAP_SEED
DISTANCE_RANGE_M = (6.0, 10.0)    # field test range, for SOP 5


# ============================================================
# DEGRADATIONS
# ============================================================

def motion_blur(image, length):
    import cv2

    length = int(round(length))
    if length <= 1:
        return image.copy()
    kernel = np.full((1, length), 1.0 / length, dtype=np.float32)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)


def _lookup(image, table):
    import cv2

    return cv2.LUT(image, np.clip(table, 0, 255).astype(np.uint8))


def low_light(image, gamma):
    levels = np.arange(256, dtype=np.float64)
    return _lookup(image, np.round(255 * (levels / 255) ** gamma))


def glare(image, alpha):
    levels = np.arange(256, dtype=np.float64)
    return _lookup(image, np.round(alpha * levels))


def occlude(image, boxes, covered):
    """
    Cover the left `covered` share of every sticker box with a dark band.
    Returns (image, visible_boxes): the new ground truth is the visible part.
    """
    result = image.copy()
    visible = []
    height, width = image.shape[:2]

    for x1, y1, x2, y2 in boxes:
        band = (x2 - x1) * covered
        cut = x1 + band
        ix1, iy1 = int(round(max(0, x1))), int(round(max(0, y1)))
        ix2, iy2 = int(round(min(width, cut))), int(round(min(height, y2)))
        if covered > 0 and ix2 > ix1 and iy2 > iy1:
            result[iy1:iy2, ix1:ix2] = OCCLUSION_COLOUR
        visible.append((cut, y1, x2, y2))

    return result, visible


def degrade(stressor, image, boxes, value):
    if stressor == "motion_blur":
        return motion_blur(image, value), list(boxes)
    if stressor == "occlusion":
        return occlude(image, boxes, value)
    if stressor == "low_light":
        return low_light(image, value), list(boxes)
    if stressor == "glare":
        return glare(image, value), list(boxes)
    raise ValueError(stressor)


# ============================================================
# MATCHING AND mAP (same rules as Ultralytics' validator)
# ============================================================

def box_iou(truth, predictions):
    """IoU matrix, truth rows x prediction columns (xyxy)."""
    if not len(truth) or not len(predictions):
        return np.zeros((len(truth), len(predictions)))
    t = np.asarray(truth, dtype=float)[:, None, :]
    p = np.asarray(predictions, dtype=float)[None, :, :]
    ix1, iy1 = np.maximum(t[..., 0], p[..., 0]), np.maximum(t[..., 1], p[..., 1])
    ix2, iy2 = np.minimum(t[..., 2], p[..., 2]), np.minimum(t[..., 3], p[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_t = (t[..., 2] - t[..., 0]) * (t[..., 3] - t[..., 1])
    area_p = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1])
    return inter / np.clip(area_t + area_p - inter, 1e-12, None)


def match_for_map(truth, predictions):
    """
    Ultralytics' greedy matching (DetectionValidator.match_predictions):
    predictions in descending confidence order; each claims its best
    unclaimed ground-truth box at every IoU threshold 0.50:0.95.
    Returns a (n_predictions, 10) boolean array.
    """
    correct = np.zeros((len(predictions), len(IOU_THRESHOLDS)), dtype=bool)
    if not len(truth) or not len(predictions):
        return correct

    iou = box_iou(truth, predictions)
    matched = np.zeros((len(truth), len(IOU_THRESHOLDS)), dtype=bool)
    columns = range(len(IOU_THRESHOLDS))

    for j in np.flatnonzero((iou >= IOU_THRESHOLDS.min()).any(0)):
        available = np.where(matched, 0, iou[:, j, None])
        k = available.argmax(0)
        correct[j] = available[k, columns] >= IOU_THRESHOLDS
        matched[k, columns] |= correct[j]

    return correct


def average_precision(tp, conf, n_truth):
    """(mAP@0.5, mAP@0.5:0.95) with Ultralytics' ap_per_class (101-point)."""
    if n_truth == 0:
        return None, None
    if len(conf) == 0:
        return 0.0, 0.0

    from ultralytics.utils.metrics import ap_per_class

    result = ap_per_class(np.asarray(tp, dtype=bool), np.asarray(conf),
                          np.zeros(len(conf)), np.zeros(n_truth))
    ap = result[5]
    if len(ap) == 0:
        return 0.0, 0.0
    return float(ap[0, 0]), float(ap[0].mean())


# ============================================================
# PREDICTION
# ============================================================

def predict(model, head, image, device):
    """All boxes down to conf 0.001, sorted by confidence (high first)."""
    result = model.predict(source=image, conf=MAP_CONF, iou=ev.NMS_IOU,
                           imgsz=ev.IMAGE_SIZE, device=device, verbose=False,
                           **head["args"])[0]
    boxes = result.boxes
    if boxes is None or not len(boxes):
        return []
    xyxy = boxes.xyxy.cpu().numpy().tolist()
    confs = boxes.conf.cpu().numpy().tolist()
    return sorted(zip(xyxy, confs), key=lambda item: -item[1])


def score_image(truth, predictions):
    """Per-sticker outcomes at 0.25 plus the pieces needed for mAP."""
    confident = [p for p in predictions if p[1] >= DETECT_CONF]
    outcomes = ef.match_stickers(truth, confident)
    detected = sum(o[0] for o in outcomes)
    return {
        "outcomes": outcomes,
        "tp": detected,
        "fp": len(confident) - detected,
        "fn": len(truth) - detected,
        "map_tp": match_for_map(truth, [p[0] for p in predictions]),
        "map_conf": [p[1] for p in predictions],
    }


# ============================================================
# DATA
# ============================================================

def load_test_set(data):
    """[(image_path, [xyxy boxes in pixels])] for labelled test images."""
    import cv2

    _yaml, paths, _names = ev.resolve_test_images(data)
    items = []
    for path in paths:
        labels = ef.read_labels(path)
        if not labels:
            continue
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"Could not read test image: {path}")
        height, width = image.shape[:2]
        boxes = [((cx - w / 2) * width, (cy - h / 2) * height,
                  (cx + w / 2) * width, (cy + h / 2) * height)
                 for cx, cy, w, h in labels]
        items.append((path, boxes))
    return items, len(paths)


def read_image(path):
    import cv2

    return cv2.imread(str(path))


# ============================================================
# SUMMARIES
# ============================================================

def summarise_step(stressor, value, model, scores, n_images):
    n_truth = sum(len(s["outcomes"]) for s in scores)
    tp = sum(s["tp"] for s in scores)
    fp = sum(s["fp"] for s in scores)
    fn = sum(s["fn"] for s in scores)
    confidences = [o[1] for s in scores for o in s["outcomes"] if o[0]]
    map_tp = (np.concatenate([s["map_tp"] for s in scores])
              if scores else np.zeros((0, 10), dtype=bool))
    map_conf = [c for s in scores for c in s["map_conf"]]
    map50, map50_95 = average_precision(map_tp, map_conf, n_truth)

    return {
        "stressor": stressor, "step_value": value, "model": model,
        "images": n_images, "stickers": n_truth,
        "detected": tp, "detection_rate": ev.safe_div(tp, n_truth),
        "mean_confidence_detected": (sum(confidences) / len(confidences)
                                     if confidences else None),
        "tp_at_0.25": tp, "fp_at_0.25": fp, "fn_at_0.25": fn,
        "precision_at_0.25": ev.safe_div(tp, tp + fp) if tp + fp else None,
        "recall_at_0.25": ev.safe_div(tp, tp + fn),
        "map50": map50, "map50_95": map50_95,
    }


def add_degradation(step_rows):
    """Degradation (%) of each metric relative to step 0 (thesis formula)."""
    baselines = {}
    for row in step_rows:
        key = (row["stressor"], row["model"])
        if key not in baselines:
            baselines[key] = row
        base = baselines[key]
        for metric in ("detection_rate", "map50", "map50_95"):
            b, v = base[metric], row[metric]
            row[f"{metric}_degradation_pct"] = (
                (b - v) / b * 100 if b and v is not None else None)
    return step_rows


def range_fraction(x50, low, high, note):
    """Where x50 falls in the tested range (0 = at baseline, 1 = at end)."""
    if x50 is not None:
        return (x50 - low) / (high - low)
    if note.startswith("detected every time"):
        return math.inf          # never broke within the tested range
    if note.startswith("never detected"):
        return 0.0
    return None


def rank_stressors(points, field_file):
    """SOP 5 ranking per model, smallest range fraction first."""
    entries = []
    for p in points:
        values = SWEEPS[p["stressor"]]["values"]
        entries.append({
            "model": p["model"], "stressor": p["stressor"],
            "x50": p["x50"], "unit": SWEEPS[p["stressor"]]["unit"],
            "tested_range": f"{values[0]:g}-{values[-1]:g}",
            "range_fraction": range_fraction(p["x50"], values[0], values[-1],
                                             p["note"]),
            "note": p["note"]})

    if Path(field_file).is_file():
        with Path(field_file).open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                x50 = float(row["x50_metres"]) if row.get("x50_metres") else None
                note = row.get("note_metres", "")
                entries.append({
                    "model": row["model"], "stressor": "distance",
                    "x50": x50, "unit": "metres (field test)",
                    "tested_range": f"{DISTANCE_RANGE_M[0]:g}-"
                                    f"{DISTANCE_RANGE_M[1]:g}",
                    "range_fraction": range_fraction(x50, *DISTANCE_RANGE_M,
                                                     note),
                    "note": note})
        distance_note = "distance included from the field test"
    else:
        distance_note = ("distance NOT included: run evaluate_field.py "
                         "first, then run this script with --rank-only")

    for model in {e["model"] for e in entries}:
        mine = [e for e in entries if e["model"] == model
                and e["range_fraction"] is not None]
        mine.sort(key=lambda e: e["range_fraction"])
        for rank, entry in enumerate(mine, start=1):
            entry["rank_most_limiting"] = rank

    for entry in entries:
        entry.setdefault("rank_most_limiting", "")
        if entry["range_fraction"] == math.inf:
            entry["range_fraction"] = "not reached"
    return entries, distance_note


# ============================================================
# CHARTS AND EXAMPLES
# ============================================================

def draw_chart(stressor, step_rows, fits, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"YOLO26n": "#2a6fb0", "YOLO11n": "#d9822b"}
    info = SWEEPS[stressor]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for name in models:
        rows = [r for r in step_rows if r["stressor"] == stressor
                and r["model"] == name]
        xs = [r["step_value"] for r in rows]
        axes[0].plot(xs, [r["detection_rate"] for r in rows], "o",
                     color=colours.get(name), label=f"{name} observed")
        fit = fits.get(name)
        if fit and fit["b0"] is not None:
            grid = np.linspace(xs[0], xs[-1], 200)
            axes[0].plot(grid, 1 / (1 + np.exp(-(fit["b0"] + fit["b1"] * grid))),
                         "-", color=colours.get(name),
                         label=f"{name} logistic fit")
        axes[1].plot(xs, [math.nan if r["map50_95"] is None else r["map50_95"]
                          for r in rows], "o-", color=colours.get(name),
                     label=name)

    axes[0].axhline(0.5, color="grey", linestyle="--", linewidth=1,
                    label="breaking point (50%)")
    axes[0].axhline(0.9, color="grey", linestyle=":", linewidth=1,
                    label="reliable detection (90%)")
    axes[0].set_ylabel(f"detection rate (conf >= {DETECT_CONF}, "
                       f"IoU >= {DETECT_IOU})")
    axes[1].set_ylabel("mAP@0.5:0.95")
    for axis in axes:
        axis.set_ylim(0, 1.05)
        axis.set_xlabel(info["unit"])
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle(f"Synthetic stressor test: {info['label']}")
    figure.tight_layout()
    path = RESULTS_DIR / f"chart_{stressor}.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def save_examples(stressor, image, boxes):
    """One sticker crop at five steps, side by side, for Chapter 5 figures."""
    import cv2

    values = SWEEPS[stressor]["values"]
    picks = sorted({values[round(i * (len(values) - 1) / 4)]
                    for i in range(5)})
    x1, y1, x2, y2 = boxes[0]
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(w, h) * 1.5
    H, W = image.shape[:2]
    left, top = int(max(0, cx - half)), int(max(0, cy - half))
    right, bottom = int(min(W, cx + half)), int(min(H, cy + half))

    tiles = []
    for value in picks:
        degraded, _ = degrade(stressor, image, boxes, value)
        tile = cv2.resize(degraded[top:bottom, left:right], (300, 300))
        cv2.putText(tile, f"{value:g}", (8, 290), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)

    path = RESULTS_DIR / "examples" / f"{stressor}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".jpg", cv2.hconcat(tiles))
    if ok:
        path.write_bytes(buffer.tobytes())
    return path


# ============================================================
# MAIN
# ============================================================

def blank_none(row):
    return {k: (round(v, 5) if isinstance(v, float) and math.isfinite(v)
                else ("" if v is None else v)) for k, v in row.items()}


def fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}"


def update_ranking():
    """
    Recompute only the SOP 5 ranking from the saved Test 3 breaking points
    and the field test's breaking distance. No model is run, so this needs
    no GPU and no dataset, and the Test 3 results themselves do not change.
    """
    path = RESULTS_DIR / "synthetic_breaking_points.csv"
    if not path.is_file():
        raise SystemExit("Run Test 3 first: synthetic_breaking_points.csv "
                         "was not found.")
    points = []
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            x50 = row.get("x50")
            points.append({"stressor": row["stressor"], "model": row["model"],
                           "x50": float(x50) if x50 not in (None, "") else None,
                           "note": row.get("note") or ""})
    ranking, distance_note = rank_stressors(points, FIELD_RESULTS)
    ev.write_csv(RESULTS_DIR / "stressor_ranking.csv",
                 [blank_none(r) for r in ranking])
    print(f"MOST LIMITING STRESSOR (SOP 5; {distance_note})")
    for entry in sorted(ranking, key=lambda e: (e["model"],
                                                e["rank_most_limiting"] or 99)):
        print(f"{entry['model']:<9} rank {entry['rank_most_limiting'] or '-'}"
              f"  {entry['stressor']:<12} x50 at "
              f"{fmt(entry['range_fraction'], 2)} of its tested range")
    print(f"\nSaved {RESULTS_DIR / 'stressor_ranking.csv'}")
    return ranking


def main(argv=None):
    parser = argparse.ArgumentParser(description="Synthetic stressor test")
    parser.add_argument("--data", default=ev.DATA)
    parser.add_argument("--stressors", nargs="+", default=list(SWEEPS),
                        choices=list(SWEEPS))
    parser.add_argument("--max-images", type=int, default=None,
                        help="use only the first N baseline images (trial)")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--allow-other-gpu", action="store_true")
    parser.add_argument("--rank-only", action="store_true",
                        help="only update the stressor ranking (after the "
                             "field test); does not run the models")
    args = parser.parse_args(argv)

    if args.rank_only:
        return update_ranking()

    import torch
    from ultralytics import YOLO

    tested_gpu = ev.check_test_gpu(args.allow_other_gpu)
    device = 0 if torch.cuda.is_available() else "cpu"
    names = list(ev.MODEL_PATHS)
    for name in names:
        if not ev.MODEL_PATHS[name].is_file():
            raise SystemExit(f"{name}: checkpoint not found: "
                             f"{ev.MODEL_PATHS[name]}")
    models = {n: (YOLO(str(ev.MODEL_PATHS[n])), ev.HEADS[n]) for n in names}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jsonl = RESULTS_DIR / "synthetic_evaluation.jsonl"
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    started = time.time()

    # ---- 1. baseline images -------------------------------------------
    print(f"Dataset: {args.data}")
    test_items, n_test = load_test_set(args.data)
    print(f"Test images: {n_test} ({len(test_items)} with a sticker)")

    baseline, baseline_rows, step0 = [], [], {}
    for index, (path, boxes) in enumerate(test_items, start=1):
        image = read_image(path)
        scores = {n: score_image(boxes, predict(m, h, image, device))
                  for n, (m, h) in models.items()}
        all_found = {n: all(o[0] for o in s["outcomes"])
                     for n, s in scores.items()}
        is_baseline = all(all_found.values())
        baseline_rows.append({
            "image": path.name, "stickers": len(boxes),
            **{f"{n}_detected_all": int(v) for n, v in all_found.items()},
            "baseline": int(is_baseline)})
        if is_baseline:
            baseline.append((path, boxes))
            step0[path.name] = scores
        print(f"\r  baseline check {index}/{len(test_items)}", end="")
    print()

    ev.write_csv(RESULTS_DIR / "baseline_images.csv", baseline_rows)
    if args.max_images:
        baseline = baseline[:args.max_images]
    if not baseline:
        raise SystemExit("No baseline images: no test image is detected by "
                         "both models. Check the models and the dataset.")
    print(f"Baseline images (both models detect every sticker): "
          f"{len(baseline)}")

    ev.append_jsonl(jsonl, {
        "record_type": "synthetic_started", "run_id": run_id,
        "data": str(args.data), "test_images": n_test,
        "baseline_images": len(baseline), "tested_gpu": tested_gpu,
        "environment": ev.environment_info(),
        "sweeps": {k: SWEEPS[k]["values"] for k in args.stressors},
        "settings": {"detect_conf": DETECT_CONF, "detect_iou": DETECT_IOU,
                     "map_conf": MAP_CONF, "nms_iou": ev.NMS_IOU,
                     "imgsz": ev.IMAGE_SIZE, "bootstrap": args.bootstrap}})

    # ---- 2. sweeps ----------------------------------------------------
    step_rows, outcome_rows = [], []
    points, comparisons = [], []

    for stressor in args.stressors:
        values = SWEEPS[stressor]["values"]
        per_step = {(v, n): [] for v in values for n in names}
        print(f"\n=== {SWEEPS[stressor]['label']}: {len(values)} steps x "
              f"{len(baseline)} images")

        for index, (path, boxes) in enumerate(baseline, start=1):
            image = read_image(path)
            if index == 1:
                save_examples(stressor, image, boxes)

            for value in values:
                if value == values[0]:              # step 0 = the original
                    scores = step0[path.name]
                else:
                    degraded, truth = degrade(stressor, image, boxes, value)
                    scores = {n: score_image(truth,
                                             predict(m, h, degraded, device))
                              for n, (m, h) in models.items()}

                for n in names:
                    per_step[(value, n)].append(scores[n])
                for s_index in range(len(boxes)):
                    row = {"stressor": stressor, "step_value": value,
                           "image": path.name, "sticker_index": s_index}
                    for n in names:
                        detected, conf, overlap = scores[n]["outcomes"][s_index]
                        row[f"{n}_detected"] = detected
                        row[f"{n}_confidence"] = conf
                        row[f"{n}_iou"] = overlap
                    outcome_rows.append(row)
            print(f"\r  image {index}/{len(baseline)}", end="")
        print()

        for value in values:
            for n in names:
                step_rows.append(summarise_step(stressor, value, n,
                                                per_step[(value, n)],
                                                len(baseline)))

        # ---- breaking points and comparison for this stressor ---------
        rows = [r for r in outcome_rows if r["stressor"] == stressor]
        x = [r["step_value"] for r in rows]
        fits = {}
        for n in names:
            fit = ef.describe_fit(x, [r[f"{n}_detected"] for r in rows],
                                  SWEEPS[stressor]["unit"])
            fits[n] = fit
            point = {"stressor": stressor, "model": n,
                     "unit": SWEEPS[stressor]["unit"], **fit}
            points.append(point)
            ev.append_jsonl(jsonl, {"record_type": "breaking_point", **point})

        first, second = names
        boot = ef.paired_bootstrap_x50(rows, (first, second), "step_value",
                                       args.bootstrap, BOOTSTRAP_SEED)
        if all(fits[n]["note"].startswith("detected every time")
               for n in names):
            boot["verdict"] = ("neither model reached its breaking point "
                               "within the tested range")
        test = ef.mcnemar_exact(rows, first, second)
        x50a, x50b = fits[first]["x50"], fits[second]["x50"]
        comparison = {
            "stressor": stressor, "comparison": f"{first} - {second}",
            "unit": SWEEPS[stressor]["unit"],
            "x50_difference": (x50a - x50b if x50a is not None
                               and x50b is not None else None),
            "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
            "bootstrap_valid_resamples": boot["valid_resamples"],
            "bootstrap_verdict": boot["verdict"],
            "paired_outcomes": len(rows),
            f"detected_only_by_{first}": test["only_" + first],
            f"detected_only_by_{second}": test["only_" + second],
            "mcnemar_exact_p": test["p_value"],
            "note": ("A larger x50 means the model still detects the sticker "
                     "at a more severe level, i.e. it tolerates the stressor "
                     "better."),
        }
        comparisons.append(comparison)
        ev.append_jsonl(jsonl, {"record_type": "comparison", **comparison})
        draw_chart(stressor, step_rows, fits, names)

    # ---- 3. save ------------------------------------------------------
    add_degradation(step_rows)
    ev.write_csv(RESULTS_DIR / "synthetic_steps.csv",
                 [blank_none(r) for r in step_rows])
    ev.write_csv(RESULTS_DIR / "synthetic_outcomes.csv",
                 [blank_none(r) for r in outcome_rows])
    ev.write_csv(RESULTS_DIR / "synthetic_breaking_points.csv",
                 [blank_none(p) for p in points])
    ev.write_csv(RESULTS_DIR / "synthetic_comparison.csv",
                 [blank_none(c) for c in comparisons])
    ranking, distance_note = rank_stressors(points, FIELD_RESULTS)
    ev.write_csv(RESULTS_DIR / "stressor_ranking.csv",
                 [blank_none(r) for r in ranking])
    ev.append_jsonl(jsonl, {"record_type": "finished", "run_id": run_id,
                            "minutes": round((time.time() - started) / 60, 1)})

    # ---- 4. printed summary -------------------------------------------
    print(f"\nBREAKING POINTS ({len(baseline)} baseline images)")
    for p in points:
        print(f"{p['stressor']:<12}{p['model']:<9} x50 = {fmt(p['x50'], 2):>7}"
              f"   x90 = {fmt(p['x90'], 2):>7}   ({p['unit']})"
              + (f"   [{p['method']}]" if p["method"] else "")
              + (f"   note: {p['note']}" if p["note"] else ""))

    print("\nCOMPARISON (YOLO26n - YOLO11n)")
    for c in comparisons:
        print(f"{c['stressor']:<12} x50 diff = {fmt(c['x50_difference'], 2)}"
              f"  95% CI [{fmt(c['ci_low'], 2)}, {fmt(c['ci_high'], 2)}]"
              f"  {c['bootstrap_verdict']}; McNemar p = "
              f"{c['mcnemar_exact_p']:.4f}")

    print(f"\nMOST LIMITING STRESSOR (SOP 5; {distance_note})")
    for entry in sorted(ranking, key=lambda e: (e["model"],
                                                e["rank_most_limiting"] or 99)):
        print(f"{entry['model']:<9} rank {entry['rank_most_limiting'] or '-'}"
              f"  {entry['stressor']:<12} x50 at "
              f"{fmt(entry['range_fraction'], 2)} of its tested range")

    print(f"\nSaved in {RESULTS_DIR}")
    return step_rows, outcome_rows, points, comparisons, ranking


if __name__ == "__main__":
    main()