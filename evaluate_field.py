"""
Chapter 5: evaluate YOLO26n and YOLO11n on the labelled field-test images.

Matches the thesis (Step 8, SOP 3 and SOP 4):
  1. Standard metrics on the field images (Ultralytics validation, same as
     evaluate_test.py): precision, recall, F1, mAP@0.5, mAP@0.5:0.95, and
     TP / FP / FN at the 0.25 threshold. Reported for all field images and
     for each distance (6-10 m, beyond the 5 m baseline).
  2. Per-sticker detection outcomes: a sticker is DETECTED when a predicted
     box has confidence >= 0.25 and IoU >= 0.5 with its ground-truth box.
  3. Distance breaking points (SOP 4), from captures of parked vehicles so
     that distance is the only condition changing: logistic regression of
     the detection outcome on sticker pixel width at the 640-pixel model
     input gives
        x50 = width at which the detection rate falls to 50% (breaking point)
        x90 = width at which it equals 90% (reliable-detection limit)
     A second regression on distance in metres gives the same two points in
     metres, which are easier to read.
  4. Model comparison on the same stickers:
        - 95% CI of the x50 difference (YOLO26n - YOLO11n) from 1,000 paired
          bootstrap resamples of the images
        - McNemar's exact test on the paired per-sticker outcomes

What you need first:
  - research_outputs/field_test/field_log.csv (from the app's Field Test tab)
  - the field images uploaded to the Ultralytics Platform as their own dataset
    and labelled (a box around every sticker). Do NOT rename the files.

Run on the RTX 2050 laptop (the script checks):
    python evaluate_field.py
    python evaluate_field.py --data ul://carl-andre-de-castro/datasets/<name>

Outputs (research_outputs/field_results/):
    field_metrics.csv          standard metrics per group and model
    field_outcomes.csv         one row per labelled sticker, both models
    field_breaking_points.csv  x50 / x90 (pixel width and metres) per model
    field_comparison.csv       bootstrap CI of the x50 difference, McNemar
    field_data_check.csv       images / log rows that could not be matched
    field_matches.csv          which labelled image each log row was linked to
    chart_distance.png         detection rate per distance + fitted curves
    field_evaluation.jsonl     everything, with versions and settings
"""

import argparse
import csv
import math
import random
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

import evaluate_test as ev


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Ultralytics Platform address of the LABELLED field dataset, or a data.yaml.
FIELD_DATA = "ul://carl-andre-de-castro/datasets/student-vehicle-sticker-field"

FIELD_LOG = BASE_DIR / "research_outputs" / "field_test" / "field_log.csv"
RESULTS_DIR = BASE_DIR / "research_outputs" / "field_results"
RUNS_DIR = BASE_DIR / "runs" / "field_eval"

# Groups reported with standard metrics: distance is the only field-test
# condition. Daylight only: dusk and night were excluded (headlights hid
# the sticker).
GROUPS = {
    "distance_m": ["6", "7", "8", "9", "10"],   # beyond the 5 m baseline
}
# (Lighting, motion and occlusion are the same for every field capture:
# daylight, parked, nothing covering the sticker. They are tested in
# Test 3 instead.)

DISTANCE_MOTION = "parked"   # distance breaking points use parked captures
DETECT_CONF = 0.25           # thesis definition of "detected"
DETECT_IOU = 0.50
BOOTSTRAP_RESAMPLES = 1000   # thesis SOP 4
BOOTSTRAP_SEED = 0
MIN_GROUP_IMAGES = 10        # smaller groups are flagged as unreliable

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# DATA
# ============================================================

def read_field_log(path):
    if not Path(path).is_file():
        raise SystemExit(f"Field log not found: {path}\n"
                         "Capture images with the app's Field Test tab first.")

    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        rows = [row for row in csv.DictReader(file) if row.get("capture_id")]

    if not rows:
        raise SystemExit(f"The field log is empty: {path}")
    return rows


def convert_field_dataset(data):
    """
    Convert the Platform dataset to YOLO folders. Ultralytics refuses a
    dataset with no 'train' split, but the field dataset is test-only (it is
    never used for training). In that case the dataset's file list is
    downloaded, a local copy with every photo marked 'train' is converted
    instead, and all splits are read anyway, so every photo is still used.
    The dataset on the Platform is not changed.
    """
    import json

    from ultralytics.data.utils import convert_ndjson_to_yolo_if_needed

    try:
        return convert_ndjson_to_yolo_if_needed(str(data), 1.0, split="test")
    except ValueError as error:
        if "'train' split" not in str(error):
            raise

    from ultralytics.utils.checks import check_file

    folder = BASE_DIR / "research_outputs" / "field_dataset_list"
    folder.mkdir(parents=True, exist_ok=True)
    listing = Path(check_file(str(data), download_dir=str(folder)))
    lines = [json.loads(line) for line in
             listing.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines[0].pop("path", None)        # keep the Platform's own image links
    for record in lines[1:]:
        record["split"] = "train"
    patched = folder / f"{listing.stem}-all.ndjson"
    patched.write_text("\n".join(json.dumps(line) for line in lines) + "\n",
                       encoding="utf-8")
    print(f"Test-only dataset: converted {len(lines) - 1} photos from a "
          "local copy of its file list")
    return convert_ndjson_to_yolo_if_needed(str(patched), 1.0)


def dataset_images(data):
    """(class_names, {image_stem: image_path}) over every dataset split."""
    from ultralytics.data.utils import check_det_dataset

    data_yaml = convert_field_dataset(data)
    config = check_det_dataset(str(data_yaml))
    root = Path(config.get("path", "."))

    images = {}
    for split in ("train", "val", "test"):
        if config.get(split):
            for path in ev.collect_images(config[split], root):
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    images[path.stem] = path

    return config.get("names", {}), images


def label_path(image_path):
    """Ultralytics' rule: replace the last 'images' folder with 'labels'."""
    parts = list(Path(image_path).parts)
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_labels(image_path):
    """YOLO label lines -> list of (cx, cy, w, h), all normalised 0-1."""
    path = label_path(image_path)
    if not path.is_file():
        return []

    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            boxes.append(tuple(float(v) for v in parts[1:5]))
    return boxes


def image_size(image_path):
    from PIL import Image

    with Image.open(image_path) as image:
        return image.size        # (width, height)


SIMILARITY_MAX = 10.0    # mean grey-level difference (0-255) for a picture match
FINGERPRINT_SIZE = 32


def logged_photo(row, log_dir):
    """The photo file on this computer that a log row refers to."""
    relative = str(row.get("image_file", "")).replace("\\", "/")
    return Path(log_dir) / relative


def log_stem(row):
    return Path(str(row.get("image_file", "")).replace("\\", "/")).stem


def content_keys(path):
    """Hashes the Ultralytics Platform may put in a downloaded file's name."""
    import hashlib

    data = Path(path).read_bytes()
    return {hashlib.md5(data).hexdigest(), hashlib.sha256(data).hexdigest(),
            hashlib.sha256(data).hexdigest()[:32],
            hashlib.sha1(data).hexdigest()[:32]}


def name_hash(image_path):
    """'022280be...13863_51' -> '022280be...13863' (the Platform's name)."""
    stem = Path(image_path).stem.lower()
    head, _, tail = stem.rpartition("_")
    return head if head and tail.isdigit() else stem


def fingerprint(path):
    """A tiny grey version of the picture, to recognise the same photo even
    after the Platform re-saved it."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        small = image.convert("L").resize((FINGERPRINT_SIZE,
                                           FINGERPRINT_SIZE))
    return np.asarray(small, dtype=float)


def match_images(rows, images, log_dir=None):
    """
    Link each log row to its labelled image, in this order:
      1. same file name
      2. identical file (the Platform renames downloads to a content hash)
      3. same picture (tiny grey fingerprint, one-to-one, closest first)
    Returns (matched, problems); every link is also listed in
    field_matches.csv so it can be checked.
    """
    log_dir = Path(log_dir) if log_dir else FIELD_LOG.parent
    by_hash = {name_hash(p): p for p in images.values()}
    links, pending, used = {}, [], set()

    for index, row in enumerate(rows):
        image = images.get(log_stem(row))
        how = "file name"
        local = logged_photo(row, log_dir)
        if image is None and local.is_file():
            image = next((by_hash[k] for k in content_keys(local)
                          if k in by_hash and by_hash[k] not in used), None)
            how = "identical file"
        if image is not None and image not in used:
            links[index] = (image, how, "")
            used.add(image)
        elif local.is_file():
            pending.append(index)

    free = [p for p in images.values() if p not in used]
    if pending and free:
        prints = {p: fingerprint(p) for p in free}
        pairs = []
        for index in pending:
            mine = fingerprint(logged_photo(rows[index], log_dir))
            for path, other in prints.items():
                pairs.append((float(abs(mine - other).mean()), index, path))
        for difference, index, path in sorted(pairs, key=lambda x: x[0]):
            if (difference <= SIMILARITY_MAX and index not in links
                    and path not in used):
                links[index] = (path, "same picture",
                                round(difference, 2))
                used.add(path)

    matched, problems, report = [], [], []
    for index, row in enumerate(rows):
        if index not in links:
            problems.append({"problem": "in log, not in labelled dataset",
                             "file": row["image_file"]})
            continue
        image, how, difference = links[index]
        report.append({"log_file": row["image_file"],
                       "distance_m": row.get("distance_m", ""),
                       "dataset_file": image.name, "matched_by": how,
                       "picture_difference": difference})
        labels = read_labels(image)
        if row.get("sticker_present") == "yes" and not labels:
            problems.append({
                "problem": "marked 'sticker present' but has no label box "
                           "(not labelled yet, or sticker not visible)",
                "file": row["image_file"]})
        if row.get("sticker_present") == "no" and labels:
            problems.append({"problem": "marked 'no sticker' but has a "
                                        "label box", "file": row["image_file"]})
        matched.append((row, image, len(labels)))

    for image in images.values():
        if image not in used:
            problems.append({"problem": "in labelled dataset, not in log "
                                        "(excluded)", "file": image.name})

    match_images.report = report
    return matched, problems


def build_groups(matched):
    """Yield (factor, level, items): all images first, then each level."""
    yield "all", "all", matched

    for factor, levels in GROUPS.items():
        seen = {row.get(factor, "") for row, _, _ in matched}
        ordered = [lvl for lvl in levels if lvl in seen]
        ordered += sorted(seen - set(levels))

        for level in ordered:
            yield factor, level, [m for m in matched
                                  if m[0].get(factor, "") == level]


def write_group_yaml(folder, name, image_paths, names):
    folder.mkdir(parents=True, exist_ok=True)
    listing = folder / f"{name}.txt"
    listing.write_text("\n".join(str(p.resolve()) for p in image_paths) + "\n",
                       encoding="utf-8")

    config = {"path": str(folder.resolve()), "train": str(listing.resolve()),
              "val": str(listing.resolve()), "test": str(listing.resolve()),
              "names": names}
    data_yaml = folder / f"{name}.yaml"
    data_yaml.write_text(yaml.safe_dump(config, sort_keys=False),
                         encoding="utf-8")
    return data_yaml


# ============================================================
# STANDARD METRICS (same Ultralytics evaluation as Chapter 4)
# ============================================================

def evaluate_group(checkpoint, head, data_yaml, device, run_name,
                   labelled_boxes):
    from ultralytics import YOLO

    common = {
        "data": str(data_yaml), "split": "test", "imgsz": ev.IMAGE_SIZE,
        "batch": ev.BATCH, "iou": ev.NMS_IOU, "device": device,
        "project": str(RUNS_DIR), "exist_ok": True, "verbose": False,
        **head["args"],
    }

    model = YOLO(str(checkpoint))
    class_id = ev.sticker_class_ids(model.names)[0]

    metrics = model.val(conf=ev.MAP_CONF, plots=True, name=run_name, **common)
    scores = ev.sticker_metrics(metrics.box, class_id)

    model = YOLO(str(checkpoint))
    at_app = model.val(conf=ev.APP_CONF, plots=True,
                       name=f"{run_name}_conf{ev.APP_CONF}", **common)
    tp, fp, fn = ev.confusion_counts(
        at_app.confusion_matrix.matrix.tolist(), class_id)

    has = labelled_boxes > 0
    p_app = ev.safe_div(tp, tp + fp) if tp + fp else None
    r_app = ev.safe_div(tp, tp + fn) if has else None

    return {
        "precision": scores["precision"] if has else None,
        "recall": scores["recall"] if has else None,
        "f1": ev.f1_score(scores["precision"], scores["recall"]) if has
        else None,
        "map50": scores["map50"] if has else None,
        "map50_95": scores["map50_95"] if has else None,
        "tp_at_app_conf": tp, "fp_at_app_conf": fp, "fn_at_app_conf": fn,
        "precision_at_app_conf": p_app,
        "recall_at_app_conf": r_app,
        "f1_at_app_conf": (ev.f1_score(p_app, r_app)
                           if p_app is not None and r_app is not None
                           else None),
        "plots_folder": str(Path(metrics.save_dir)),
    }


# ============================================================
# PER-STICKER OUTCOMES
# ============================================================

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
             - inter)
    return inter / union if union > 0 else 0.0


def match_stickers(truth, predictions):
    """
    Greedy one-to-one matching, highest confidence first.
    truth: list of xyxy. predictions: list of (xyxy, conf), conf >= 0.25.
    Returns, per truth box, (detected, confidence, iou).
    """
    results = [(0, None, 0.0)] * len(truth)
    taken = set()

    for box, conf in sorted(predictions, key=lambda p: -p[1]):
        best, best_iou = None, DETECT_IOU
        for index, gt in enumerate(truth):
            if index in taken:
                continue
            value = iou(box, gt)
            if value >= best_iou:
                best, best_iou = index, value
        if best is not None:
            taken.add(best)
            results[best] = (1, conf, best_iou)

    return results


def sticker_outcomes(models, matched, device):
    """One row per labelled sticker with both models' outcomes."""
    rows = []

    for row, image, _ in matched:
        labels = read_labels(image)
        if not labels:
            continue

        width, height = image_size(image)
        scale = ev.IMAGE_SIZE / max(width, height)
        truth = [((cx - w / 2) * width, (cy - h / 2) * height,
                  (cx + w / 2) * width, (cy + h / 2) * height)
                 for cx, cy, w, h in labels]

        outcome = {}
        for name, (model, head) in models.items():
            result = model.predict(
                source=str(image), conf=DETECT_CONF, iou=ev.NMS_IOU,
                imgsz=ev.IMAGE_SIZE, device=device, verbose=False,
                **head["args"])[0]
            boxes = result.boxes
            predictions = []
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy().tolist()
                confs = boxes.conf.cpu().numpy().tolist()
                predictions = list(zip(xyxy, confs))
            outcome[name] = match_stickers(truth, predictions)

        for index, (cx, cy, w, h) in enumerate(labels):
            item = {
                "image": image.name,
                "capture_id": row.get("capture_id", ""),
                "sticker_index": index,
                **{factor: row.get(factor, "") for factor in GROUPS},
                "motion": row.get("motion", "") or DISTANCE_MOTION,
                "sticker_px_width_640": w * width * scale,
                "sticker_px_height_640": h * height * scale,
            }
            for name in models:
                detected, conf, overlap = outcome[name][index]
                item[f"{name}_detected"] = detected
                item[f"{name}_confidence"] = conf
                item[f"{name}_iou"] = overlap
            rows.append(item)

    return rows


# ============================================================
# LOGISTIC REGRESSION, BOOTSTRAP, McNEMAR
# ============================================================

def fit_logistic(x, y, iterations=100):
    """
    Fit p(detected) = 1 / (1 + exp(-(b0 + b1 x))) by Newton-Raphson.
    Returns (b0, b1) or None when the fit is not possible (all outcomes
    the same, no spread in x, or perfect separation).
    """
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3 or len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return None

    mean, sd = x.mean(), x.std() or 1.0
    z = (x - mean) / sd                     # standardise for stability
    a0 = a1 = 0.0

    for _ in range(iterations):
        p = 1 / (1 + np.exp(-np.clip(a0 + a1 * z, -35, 35)))
        weight = p * (1 - p)
        g0, g1 = (y - p).sum(), ((y - p) * z).sum()
        h00, h01, h11 = weight.sum(), (weight * z).sum(), (weight * z * z).sum()
        det = h00 * h11 - h01 * h01
        if det <= 1e-12:
            return None
        step0 = (h11 * g0 - h01 * g1) / det
        step1 = (h00 * g1 - h01 * g0) / det
        a0 += step0
        a1 += step1
        if abs(a1) > 50:                    # perfect separation
            return None
        if abs(step0) < 1e-10 and abs(step1) < 1e-10:
            break

    return float(a0 - a1 * mean / sd), float(a1 / sd)


def breaking_values(coefficients):
    """x50 = -b0/b1, x90 = (ln 9 - b0)/b1 (thesis Chapter III)."""
    if coefficients is None or abs(coefficients[1]) < 1e-12:
        return None, None
    b0, b1 = coefficients
    return -b0 / b1, (math.log(9) - b0) / b1


MAX_LEVELS_FOR_INTERPOLATION = 25   # interpolation only for stepped levels


def interpolate_point(x, y, target):
    """
    Fallback when logistic regression is not possible (complete separation):
    the level at which the observed detection rate crosses `target`, by
    linear interpolation between the two neighbouring tested levels.
    Levels are ordered from the easiest condition to the hardest.
    """
    levels = sorted(set(x))
    rates = [sum(yi for xi, yi in zip(x, y) if xi == level)
             / sum(1 for xi in x if xi == level) for level in levels]
    if rates[0] < rates[-1]:                 # easiest condition is high x
        levels, rates = levels[::-1], rates[::-1]
    if rates[0] < target:
        return levels[0]                     # already below at the easiest
    for i in range(1, len(levels)):
        if rates[i] < target:
            l0, l1, r0, r1 = levels[i - 1], levels[i], rates[i - 1], rates[i]
            return l0 + (r0 - target) / (r0 - r1) * (l1 - l0)
    return None                              # never falls below target


WRONG_DIRECTION = "wrong direction"


def x50_x90(x, y, harder="higher"):
    """
    (x50, x90, method): logistic regression, else interpolation.
    harder: "higher" when a larger x is a harder condition (metres, blur,
    gamma...), "lower" when a smaller x is harder (sticker pixel width).
    If detection does NOT fall as the condition gets harder, there is no
    breaking point: (None, None, WRONG_DIRECTION).
    """
    coefficients = fit_logistic(x, y)
    if coefficients is not None:
        slope = coefficients[1]
        if (harder == "higher" and slope >= 0) or (harder == "lower"
                                                   and slope <= 0):
            return None, None, WRONG_DIRECTION
        x50, x90 = breaking_values(coefficients)
        return x50, x90, "logistic regression"
    if (y and 0 < sum(y) < len(y)
            and 2 <= len(set(x)) <= MAX_LEVELS_FOR_INTERPOLATION):
        return (interpolate_point(x, y, 0.5), interpolate_point(x, y, 0.9),
                "linear interpolation (complete separation)")
    return None, None, ""


def describe_fit(x, y, label, harder="higher"):
    coefficients = fit_logistic(x, y)
    x50, x90, method = x50_x90(x, y, harder)
    rate = sum(y) / len(y) if y else None

    note = ""
    if method == WRONG_DIRECTION:
        method = ""
        note = ("not estimable: detection did not fall as the condition got "
                "harder (check the sample size per level)")
    elif not method:
        if y and all(y):
            note = "detected every time: limit not reached in tested range"
        elif y and not any(y):
            note = "never detected in tested range"
        else:
            note = "not estimable (too little variation or separation)"
    else:
        low, high = min(x), max(x)
        outside = [name for name, v in (("x50", x50), ("x90", x90))
                   if v is not None and not low <= v <= high]
        missing = [name for name, v in (("x50", x50), ("x90", x90))
                   if v is None]
        parts = []
        if outside:
            parts.append(f"{' and '.join(outside)} outside the tested range "
                         f"({low:g}-{high:g} {label}); extrapolated")
        if missing:
            parts.append(f"{' and '.join(missing)} not reached in tested "
                         "range")
        note = "; ".join(parts)

    return {"b0": coefficients[0] if coefficients else None,
            "b1": coefficients[1] if coefficients else None,
            "x50": x50, "x90": x90, "method": method,
            "detection_rate": rate, "stickers": len(y), "note": note}


def paired_bootstrap_x50(outcomes, models, x_key, resamples, seed,
                         harder="higher"):
    """
    Resample IMAGES with replacement (both models keep the same images) and
    refit each model; returns the 95% percentile CI of x50(A) - x50(B).
    """
    first, second = models
    by_image = {}
    for row in outcomes:
        by_image.setdefault(row["image"], []).append(row)
    images = list(by_image)
    rng = random.Random(seed)
    differences = []

    for _ in range(resamples):
        sample = [r for _ in images for r in by_image[rng.choice(images)]]
        x = [r[x_key] for r in sample]
        a = x50_x90(x, [r[f"{first}_detected"] for r in sample], harder)[0]
        b = x50_x90(x, [r[f"{second}_detected"] for r in sample], harder)[0]
        if a is not None and b is not None:
            differences.append(a - b)

    if len(differences) < resamples * 0.5:
        return {"valid_resamples": len(differences), "ci_low": None,
                "ci_high": None, "verdict": "not estimable (too many "
                "resamples without a valid fit)"}

    low = ev.percentile(differences, 2.5)
    high = ev.percentile(differences, 97.5)
    verdict = ("inconclusive (interval includes zero)"
               if low <= 0 <= high else "difference is clear")
    return {"valid_resamples": len(differences), "ci_low": low,
            "ci_high": high, "verdict": verdict}


def mcnemar_exact(outcomes, first, second):
    """Exact two-sided McNemar test on paired per-sticker outcomes."""
    only_first = sum(1 for r in outcomes if r[f"{first}_detected"]
                     and not r[f"{second}_detected"])
    only_second = sum(1 for r in outcomes if r[f"{second}_detected"]
                      and not r[f"{first}_detected"])
    n = only_first + only_second

    if n == 0:
        p_value = 1.0
    else:
        k = min(only_first, only_second)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
        p_value = min(1.0, 2 * tail)

    return {"only_" + first: only_first, "only_" + second: only_second,
            "p_value": p_value}


# ============================================================
# CHART
# ============================================================

def draw_distance_chart(outcomes, fits_m, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"YOLO26n": "#2a6fb0", "YOLO11n": "#d9822b"}
    distances = sorted({float(r["distance_m"]) for r in outcomes
                        if r["distance_m"]})
    if not distances:
        return None

    figure, axis = plt.subplots(figsize=(7, 4.2))
    for name in models:
        rates = []
        for d in distances:
            group = [r[f"{name}_detected"] for r in outcomes
                     if r["distance_m"] and float(r["distance_m"]) == d]
            rates.append(sum(group) / len(group) if group else math.nan)
        axis.plot(distances, rates, "o", color=colours.get(name),
                  label=f"{name} observed")

        fit = fits_m.get(name)
        if fit and fit["b0"] is not None:
            xs = [distances[0] + i * (distances[-1] - distances[0]) / 100
                  for i in range(101)]
            ys = [1 / (1 + math.exp(-(fit["b0"] + fit["b1"] * x)))
                  for x in xs]
            axis.plot(xs, ys, "-", color=colours.get(name),
                      label=f"{name} logistic fit")

    axis.axhline(0.5, color="grey", linestyle="--", linewidth=1,
                 label="breaking point (50%)")
    axis.axhline(0.9, color="grey", linestyle=":", linewidth=1,
                 label="reliable detection (90%)")
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("distance (m)")
    axis.set_ylabel(f"detection rate (conf >= {DETECT_CONF}, "
                    f"IoU >= {DETECT_IOU})")
    axis.set_title("Field test: detection rate by distance")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = RESULTS_DIR / "chart_distance.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


# ============================================================
# MAIN
# ============================================================

def blank_none(row):
    return {k: (round(v, 5) if isinstance(v, float) else
                ("" if v is None else v)) for k, v in row.items()}


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Chapter 5 field evaluation")
    parser.add_argument("--data", default=FIELD_DATA)
    parser.add_argument("--log", default=str(FIELD_LOG))
    parser.add_argument("--allow-other-gpu", action="store_true")
    args = parser.parse_args(argv)

    import torch
    from ultralytics import YOLO

    tested_gpu = ev.check_test_gpu(args.allow_other_gpu,
                                   expected=ev.EXPECTED_GPU)
    model_names = list(ev.MODEL_PATHS)

    rows = read_field_log(args.log)
    print(f"Field log: {len(rows)} captures")
    names, images = dataset_images(args.data)
    print(f"Labelled images found: {len(images)}")
    matched, problems = match_images(rows, images, Path(args.log).parent)
    print(f"Matched images: {len(matched)}")
    ways = {}
    for item in match_images.report:
        ways[item["matched_by"]] = ways.get(item["matched_by"], 0) + 1
    if ways:
        print("  matched by " + ", ".join(f"{k}: {v}" for k, v in ways.items()))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ev.write_csv(RESULTS_DIR / "field_data_check.csv",
                 problems or [{"problem": "none", "file": ""}])
    ev.write_csv(RESULTS_DIR / "field_matches.csv",
                 match_images.report or [{"log_file": "none"}])
    if problems:
        print(f"WARNING: {len(problems)} data problem(s). "
              "See field_data_check.csv")
    if not matched:
        raise SystemExit("No field images could be matched to the log. "
                         "Were the files renamed?")

    for name in model_names:
        if not ev.MODEL_PATHS[name].is_file():
            raise SystemExit(f"{name}: checkpoint not found: "
                             f"{ev.MODEL_PATHS[name]}")

    device = 0 if torch.cuda.is_available() else "cpu"
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    jsonl = RESULTS_DIR / "field_evaluation.jsonl"
    ev.append_jsonl(jsonl, {
        "record_type": "field_evaluation_started", "run_id": run_id,
        "data": str(args.data), "captures": len(rows),
        "matched_images": len(matched), "problems": len(problems),
        "tested_gpu": tested_gpu, "environment": ev.environment_info(),
        "settings": {"detect_conf": DETECT_CONF, "detect_iou": DETECT_IOU,
                     "map_conf": ev.MAP_CONF, "nms_iou": ev.NMS_IOU,
                     "imgsz": ev.IMAGE_SIZE,
                     "bootstrap_resamples": BOOTSTRAP_RESAMPLES},
    })

    # ---- 1. standard metrics per group -------------------------------
    metric_rows = []
    with tempfile.TemporaryDirectory() as temp:
        for factor, level, items in build_groups(matched):
            group = f"{factor}_{level}".replace(" ", "_")
            boxes = sum(count for _, _, count in items)
            data_yaml = write_group_yaml(
                Path(temp), group, [image for _, image, _ in items], names)

            for name in model_names:
                print(f"--- metrics: {name}, {factor} = {level} "
                      f"({len(items)} images, {boxes} stickers)")
                result = evaluate_group(ev.MODEL_PATHS[name], ev.HEADS[name],
                                        data_yaml, device, f"{group}/{name}",
                                        boxes)
                row = {"group": factor, "level": level, "model": name,
                       "inference_head": ev.HEADS[name]["label"],
                       "images": len(items), "labelled_stickers": boxes,
                       "small_group": len(items) < MIN_GROUP_IMAGES,
                       **result, "run_id": run_id}
                metric_rows.append(row)
                ev.append_jsonl(jsonl, {"record_type": "field_group", **row})

    ev.write_csv(RESULTS_DIR / "field_metrics.csv",
                 [blank_none(r) for r in metric_rows])

    # ---- 2. per-sticker outcomes --------------------------------------
    print("\n--- per-sticker outcomes (conf >= 0.25, IoU >= 0.5)")
    models = {name: (YOLO(str(ev.MODEL_PATHS[name])), ev.HEADS[name])
              for name in model_names}
    outcomes = sticker_outcomes(models, matched, device)
    ev.write_csv(RESULTS_DIR / "field_outcomes.csv",
                 [blank_none(r) for r in outcomes] or [{"image": "none"}])
    if not outcomes:
        raise SystemExit("No labelled stickers found in the field images.")

    # ---- 3. breaking points (SOP 4) -----------------------------------
    # Thesis: the vehicle is stationary at each marker so that distance is
    # the only condition that changes. Captures marked as moving are kept in
    # the metrics above but left out of the distance breaking points.
    parked = [r for r in outcomes if r.get("motion", "parked") in
              ("", DISTANCE_MOTION)]
    if not parked:
        raise SystemExit("No parked captures with stickers: the distance "
                         "breaking points need captures marked 'parked'.")
    width = [r["sticker_px_width_640"] for r in parked]
    with_distance = [r for r in parked if r["distance_m"]]
    metres = [float(r["distance_m"]) for r in with_distance]

    points, fits_m = [], {}
    for name in model_names:
        by_width = describe_fit(width, [r[f"{name}_detected"]
                                        for r in parked], "px",
                                harder="lower")
        by_metre = describe_fit(metres, [r[f"{name}_detected"]
                                         for r in with_distance], "m")
        fits_m[name] = by_metre
        point = {
            "model": name, "stickers": by_width["stickers"],
            "detection_rate": by_width["detection_rate"],
            "x50_px_width": by_width["x50"], "x90_px_width": by_width["x90"],
            "note_px": by_width["note"],
            "method_px": by_width["method"],
            "x50_metres": by_metre["x50"], "x90_metres": by_metre["x90"],
            "note_metres": by_metre["note"],
            "method_metres": by_metre["method"],
            "b0_px": by_width["b0"], "b1_px": by_width["b1"],
            "b0_m": by_metre["b0"], "b1_m": by_metre["b1"],
        }
        points.append(point)
        ev.append_jsonl(jsonl, {"record_type": "breaking_point", **point})

    ev.write_csv(RESULTS_DIR / "field_breaking_points.csv",
                 [blank_none(p) for p in points])

    # ---- 4. model comparison ------------------------------------------
    first, second = model_names[0], model_names[1]
    print(f"--- paired bootstrap ({BOOTSTRAP_RESAMPLES} resamples)")
    boot_px = paired_bootstrap_x50(parked, (first, second),
                                   "sticker_px_width_640",
                                   BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED,
                                   harder="lower")
    test = mcnemar_exact(parked, first, second)
    comparison = {
        "comparison": f"{first} - {second}",
        "x50_px_difference": (
            points[0]["x50_px_width"] - points[1]["x50_px_width"]
            if points[0]["x50_px_width"] is not None
            and points[1]["x50_px_width"] is not None else None),
        "x50_px_diff_ci_low": boot_px["ci_low"],
        "x50_px_diff_ci_high": boot_px["ci_high"],
        "bootstrap_valid_resamples": boot_px["valid_resamples"],
        "bootstrap_verdict": boot_px["verdict"],
        "stickers": len(parked),
        f"detected_only_by_{first}": test["only_" + first],
        f"detected_only_by_{second}": test["only_" + second],
        "mcnemar_exact_p": test["p_value"],
        "note": ("A smaller x50 pixel width means the model still detects "
                 "smaller (farther) stickers, i.e. it tolerates distance "
                 "better."),
    }
    ev.write_csv(RESULTS_DIR / "field_comparison.csv",
                 [blank_none(comparison)])
    ev.append_jsonl(jsonl, {"record_type": "comparison", **comparison})

    chart = draw_distance_chart(parked, fits_m, model_names)

    # ---- printed summary ----------------------------------------------
    print(f"\nFIELD METRICS (* = fewer than {MIN_GROUP_IMAGES} images)")
    print(f"{'group':<11}{'level':<14}{'model':<9}{'imgs':>5}{'stk':>5}"
          f"{'P@.25':>8}{'R@.25':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for r in metric_rows:
        flag = "*" if r["small_group"] else " "
        print(f"{r['group']:<11}{r['level'] + flag:<14}{r['model']:<9}"
              f"{r['images']:>5}{r['labelled_stickers']:>5}"
              f"{fmt(r['precision_at_app_conf']):>8}"
              f"{fmt(r['recall_at_app_conf']):>8}{fmt(r['map50']):>8}"
              f"{fmt(r['map50_95']):>10}")

    print("\nDISTANCE BREAKING POINTS (logistic regression)")
    for p in points:
        print(f"{p['model']:<9} x50 = {fmt(p['x50_px_width'], 1)} px "
              f"({fmt(p['x50_metres'], 2)} m)   x90 = "
              f"{fmt(p['x90_px_width'], 1)} px "
              f"({fmt(p['x90_metres'], 2)} m)")
        for label, note in (("pixel width", p["note_px"]),
                            ("metres", p["note_metres"])):
            if note:
                print(f"          note ({label}): {note}")

    print(f"\nCOMPARISON {first} vs {second}")
    print(f"x50 difference: {fmt(comparison['x50_px_difference'], 2)} px, "
          f"95% CI [{fmt(boot_px['ci_low'], 2)}, "
          f"{fmt(boot_px['ci_high'], 2)}] -> {boot_px['verdict']}")
    print(f"McNemar: only {first} detected {test['only_' + first]}, only "
          f"{second} detected {test['only_' + second]}, "
          f"exact p = {test['p_value']:.4f}")

    print(f"\nSaved in {RESULTS_DIR}")
    if chart:
        print(f"Chart: {chart.name}")
    return metric_rows, outcomes, points, comparison


if __name__ == "__main__":
    main()