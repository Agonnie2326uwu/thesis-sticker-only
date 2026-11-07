"""
Photo gallery for the dashboard's Photos page.

Runs both models on the photos used in each test and saves one picture per
photo with YOLO26n's boxes on the left and YOLO11n's boxes on the right, so
the panel can see exactly which stickers each model found or missed.

  Test 1  every photo of the 210-photo test set
  Test 2  every labelled field photo (needs the field log and field dataset)
  Test 3  a few baseline photos at every stressor level
          (needs Test 3's baseline_images.csv)

Same rules as the tests: a sticker counts as found when a box has
confidence >= 0.25 and IoU >= 0.50 with the labelled box (greedy matching,
highest confidence first). YOLO26n runs NMS-free, YOLO11n with NMS.

This only makes pictures for viewing. It does not change any test result,
so it can run on any computer and needs no GPU check.

Privacy: the models run on the same photos as the tests (the original
dataset copy), but the pictures SHOWN are drawn on the blurred copy on the
Ultralytics Platform (plates and faces blurred), matched by file name.
A photo with no blurred copy is left out of the gallery and listed, so no
unblurred plate is shown to the panel.

Usage (from the project folder, with the venv active):
    python make_gallery.py
    python make_gallery.py --sets test synthetic --synthetic-photos 8
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import evaluate_field as ef
import evaluate_synthetic as es
import evaluate_test as ev


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
GALLERY_DIR = BASE_DIR / "research_outputs" / "gallery"
INDEX = GALLERY_DIR / "gallery_index.csv"
SYNTHETIC_BASELINE = (BASE_DIR / "research_outputs" / "synthetic_results"
                      / "baseline_images.csv")

SETS = ("test", "field", "synthetic")
SYNTHETIC_PHOTOS = 8            # baseline photos shown at every level
HALF_WIDTH = 640                # width of each model's half of the picture
HEADER = 38                     # height of the title bar above each half
GAP = 10                        # white gap between the two halves

# Same colours as the dashboard charts (OpenCV uses BGR order).
COLOURS = {"YOLO26n": (214, 120, 42), "YOLO11n": (52, 104, 235)}
MISSED = (40, 40, 220)          # red box for a sticker the model missed
DARK = (24, 24, 24)
WHITE = (255, 255, 255)


# ============================================================
# MATCHING (same rule as the tests)
# ============================================================

def match(truth, predictions):
    """
    Greedy one-to-one matching, highest confidence first, IoU >= 0.50.
    truth: list of xyxy. predictions: list of (xyxy, conf), conf >= 0.25.
    Returns (set of found truth indexes, set of matched prediction indexes).
    """
    found, used = set(), set()
    order = sorted(range(len(predictions)), key=lambda j: -predictions[j][1])

    for j in order:
        box = predictions[j][0]
        best, best_iou = None, ef.DETECT_IOU
        for i, gt in enumerate(truth):
            if i in found:
                continue
            value = ef.iou(box, gt)
            if value >= best_iou:
                best, best_iou = i, value
        if best is not None:
            found.add(best)
            used.add(j)

    return found, used


def confident(model, head, image, device):
    """The model's boxes at the thesis threshold (conf >= 0.25)."""
    return [p for p in es.predict(model, head, image, device)
            if p[1] >= ef.DETECT_CONF]


def verdict(stickers, found):
    """Short result label used by the dashboard's filter."""
    a, b = found["YOLO26n"], found["YOLO11n"]
    if stickers == 0:
        return "no sticker"
    if a == stickers and b == stickers:
        return "both found all"
    if a == stickers:
        return "only YOLO26n found all"
    if b == stickers:
        return "only YOLO11n found all"
    return "both missed some"


# ============================================================
# DRAWING
# ============================================================

def label(image, text, x, y, colour, scale=0.55):
    """Text on a filled box, kept inside the picture."""
    import cv2

    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    height, width = image.shape[:2]
    x = int(min(max(0, x), max(0, width - w - 6)))
    y = int(max(h + base + 4, y))
    cv2.rectangle(image, (x, y - h - base - 4), (x + w + 6, y), colour, -1)
    cv2.putText(image, text, (x + 3, y - base - 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, WHITE, 1, cv2.LINE_AA)


def draw_half(image, name, truth, predictions):
    """One model's view: its boxes, its misses and a title bar."""
    import cv2
    import numpy as np

    found, used = match(truth, predictions)
    scale = HALF_WIDTH / image.shape[1]
    picture = cv2.resize(image, (HALF_WIDTH,
                                 max(1, int(round(image.shape[0] * scale)))))
    colour = COLOURS[name]

    def pixels(box):
        return tuple(int(round(v * scale)) for v in box)

    for j, (box, conf) in enumerate(predictions):
        x1, y1, x2, y2 = pixels(box)
        if j in used:
            cv2.rectangle(picture, (x1, y1), (x2, y2), colour, 3)
            label(picture, f"sticker {conf:.2f}", x1, y1 - 2, colour)
        else:
            cv2.rectangle(picture, (x1, y1), (x2, y2), colour, 2)
            label(picture, f"WRONG {conf:.2f}", x1, y1 - 2, colour)

    for i, box in enumerate(truth):
        if i not in found:
            x1, y1, x2, y2 = pixels(box)
            cv2.rectangle(picture, (x1, y1), (x2, y2), MISSED, 3)
            label(picture, "MISSED", x1, y2 + 22, MISSED)

    wrong = len(predictions) - len(used)
    bar = np.full((HEADER, HALF_WIDTH, 3), DARK, dtype=np.uint8)
    cv2.rectangle(bar, (12, 11), (28, 27), colour, -1)
    summary = (f"{name}   found {len(found)} of {len(truth)}   "
               f"wrong {wrong}" if truth else
               f"{name}   no sticker in photo   wrong {wrong}")
    cv2.putText(bar, summary, (38, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                WHITE, 1, cv2.LINE_AA)

    return np.vstack([bar, picture]), len(found), wrong


def side_by_side(image, truth, predictions_by_model, display=None):
    """
    Both halves next to each other, plus the counts for the index.
    `image` is what the models saw; `display` (same size) is what is drawn
    on, e.g. the blurred copy of the same photo.
    """
    import numpy as np

    shown = image if display is None else display
    halves, counts = [], {}
    for name in ev.MODEL_PATHS:
        half, found, wrong = draw_half(shown, name, truth,
                                       predictions_by_model[name])
        halves.append(half)
        counts[name] = (found, wrong)

    gap = np.full((halves[0].shape[0], GAP, 3), WHITE, dtype=np.uint8)
    return np.hstack([halves[0], gap, halves[1]]), counts


def save(picture, path):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        path.write_bytes(buffer.tobytes())


def pixel_boxes(labels, width, height):
    return [((cx - w / 2) * width, (cy - h / 2) * height,
             (cx + w / 2) * width, (cy + h / 2) * height)
            for cx, cy, w, h in labels]


def index_row(test, image_name, file, truth, counts, **extra):
    found = {n: counts[n][0] for n in counts}
    row = {"set": test, "image": image_name,
           "file": file.relative_to(GALLERY_DIR).as_posix(),
           "stressor": "", "level": "", "distance_m": "", "lighting": "",
           "stickers": len(truth)}
    row.update(extra)
    for name, (f, w) in counts.items():
        row[f"{name}_found"] = f
        row[f"{name}_wrong"] = w
    row["result"] = verdict(len(truth), found)
    return row


# ============================================================
# BLURRED COPY FOR DISPLAY
# ============================================================

SKIPPED = []          # photos left out because no blurred copy was found


def blurred_copies(data):
    """{file stem: path} of the blurred dataset on the Platform, or None
    when the tests already use the Platform copy (nothing to swap)."""
    if str(data) == str(ev.PLATFORM_DATA):
        return None
    try:
        _names, images = ef.dataset_images(ev.PLATFORM_DATA)
    except Exception as error:      # no API key / offline
        raise SystemExit("Could not load the blurred copy from the "
                         f"Ultralytics Platform ({error}). Pictures are not "
                         "made, so no unblurred photo is shown.")
    print(f"  Blurred copy for display: {len(images)} photos")
    return images


def display_image(image, path, blurred):
    """The blurred copy of `path`, resized to `image`, or None to skip."""
    import cv2

    if blurred is None:
        return image
    match = blurred.get(Path(path).stem)
    shown = cv2.imread(str(match)) if match else None
    if shown is None:
        SKIPPED.append(Path(path).name)
        return None
    if shown.shape[:2] != image.shape[:2]:
        shown = cv2.resize(shown, (image.shape[1], image.shape[0]))
    return shown


# ============================================================
# THE THREE SETS
# ============================================================

def fresh(key):
    """Empty one set's folder, once its inputs have loaded."""
    shutil.rmtree(GALLERY_DIR / key, ignore_errors=True)


def make_test(models, device, data):
    import cv2

    _yaml, paths, _names = ev.resolve_test_images(data)
    blurred = blurred_copies(data)
    fresh("test")
    rows = []
    for number, path in enumerate(paths, start=1):
        image = cv2.imread(str(path))
        if image is None:
            continue
        shown = display_image(image, path, blurred)
        if shown is None:
            continue
        height, width = image.shape[:2]
        truth = pixel_boxes(ef.read_labels(path), width, height)
        predictions = {n: confident(m, h, image, device)
                       for n, (m, h) in models.items()}
        picture, counts = side_by_side(image, truth, predictions, shown)
        file = GALLERY_DIR / "test" / f"{path.stem}.jpg"
        save(picture, file)
        rows.append(index_row("test", path.name, file, truth, counts))
        print(f"\r  Test 1 photos {number}/{len(paths)}", end="")
    print()
    return rows


def make_field(models, device, data, log):
    import cv2

    log_rows = ef.read_field_log(log)
    _names, images = ef.dataset_images(data)
    matched, _problems = ef.match_images(log_rows, images)
    if not matched:
        raise SystemExit("No field photo in the log is in the labelled "
                         "field dataset yet.")
    fresh("field")

    rows = []
    for number, (row, path, _n) in enumerate(matched, start=1):
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        truth = pixel_boxes(ef.read_labels(path), width, height)
        predictions = {n: confident(m, h, image, device)
                       for n, (m, h) in models.items()}
        picture, counts = side_by_side(image, truth, predictions)
        file = GALLERY_DIR / "field" / f"{path.stem}.jpg"
        save(picture, file)
        rows.append(index_row("field", path.name, file, truth, counts,
                              distance_m=row.get("distance_m", ""),
                              lighting=row.get("lighting", "")))
        print(f"\r  Field photos {number}/{len(matched)}", end="")
    print()
    return rows


def baseline_names(path):
    if not Path(path).is_file():
        raise SystemExit("Run Test 3 first: its list of baseline photos "
                         f"({Path(path).name}) was not found.")
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        return [r["image"] for r in csv.DictReader(file)
                if str(r.get("baseline")) == "1"]


def make_synthetic(models, device, data, photos):
    import cv2

    wanted = baseline_names(SYNTHETIC_BASELINE)[:photos]
    if not wanted:
        raise SystemExit("Test 3 found no baseline photos.")
    items, _n = es.load_test_set(data)
    blurred = blurred_copies(data)
    by_name = {path.name: (path, boxes) for path, boxes in items}
    chosen = [by_name[n] for n in wanted if n in by_name]

    fresh("synthetic")
    total = len(chosen) * sum(len(s["values"]) for s in es.SWEEPS.values())
    rows, done = [], 0
    for path, boxes in chosen:
        image = cv2.imread(str(path))
        if image is None:
            continue
        shown = display_image(image, path, blurred)
        if shown is None:
            continue
        for stressor, sweep in es.SWEEPS.items():
            for step, value in enumerate(sweep["values"]):
                degraded, truth = es.degrade(stressor, image, boxes, value)
                shown_degraded, _ = es.degrade(stressor, shown, boxes, value)
                predictions = {n: confident(m, h, degraded, device)
                               for n, (m, h) in models.items()}
                picture, counts = side_by_side(degraded, truth, predictions,
                                               shown_degraded)
                file = (GALLERY_DIR / "synthetic" / stressor
                        / f"{path.stem}_{step:02d}.jpg")
                save(picture, file)
                rows.append(index_row("synthetic", path.name, file, truth,
                                      counts, stressor=stressor,
                                      level=f"{value:g}"))
                done += 1
                print(f"\r  Test 3 photos {done}/{total}", end="")
    print()
    return rows


# ============================================================
# MAIN
# ============================================================

def read_index():
    if not INDEX.is_file():
        return []
    with INDEX.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Photos for the dashboard")
    parser.add_argument("--sets", nargs="+", default=list(SETS), choices=SETS)
    parser.add_argument("--data", default=ev.DATA)
    parser.add_argument("--field-data", default=ef.FIELD_DATA)
    parser.add_argument("--log", default=str(ef.FIELD_LOG))
    parser.add_argument("--synthetic-photos", type=int,
                        default=SYNTHETIC_PHOTOS)
    args = parser.parse_args(argv)

    import torch
    from ultralytics import YOLO

    device = 0 if torch.cuda.is_available() else "cpu"
    for name, path in ev.MODEL_PATHS.items():
        if not path.is_file():
            raise SystemExit(f"{name}: checkpoint not found: {path}")
    models = {n: (YOLO(str(p)), ev.HEADS[n]) for n, p in ev.MODEL_PATHS.items()}

    makers = {
        "test": lambda: make_test(models, device, args.data),
        "field": lambda: make_field(models, device, args.field_data, args.log),
        "synthetic": lambda: make_synthetic(models, device, args.data,
                                            args.synthetic_photos),
    }
    names = {"test": "Test 1", "field": "Test 2 (field)",
             "synthetic": "Test 3"}

    old = read_index()
    made, skipped = [], []
    for key in args.sets:
        print(f"\n=== {names[key]} photos ===")
        try:
            made += makers[key]()
        except SystemExit as reason:
            print(f"  Skipped: {reason}")
            skipped.append(key)
        except Exception as error:          # e.g. field dataset not online
            print(f"  Skipped: {type(error).__name__}: {error}")
            skipped.append(key)

    if not made and skipped == list(args.sets):
        raise SystemExit("No photos were made. See the messages above.")

    remade = set(args.sets) - set(skipped)
    kept = [r for r in old if r.get("set") not in remade
            and (GALLERY_DIR / r.get("file", "")).is_file()]

    ev.write_csv(INDEX, kept + made)
    print(f"\nSaved {len(made)} pictures in {GALLERY_DIR}")
    if SKIPPED:
        print(f"Left out {len(SKIPPED)} photos with no blurred copy on the "
              "Platform (renamed or missing there), so no unblurred plate is "
              "shown:")
        for name in sorted(set(SKIPPED))[:40]:
            print(f"  {name}")
    if skipped:
        print("Skipped: " + ", ".join(names[k] for k in skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())