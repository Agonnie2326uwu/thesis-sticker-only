"""
PSU Vehicle Sticker Detection - live research demonstrator.

This app shows YOLO26n and YOLO11n running on a webcam feed. It is a
demonstration tool only. Webcam observations are NOT the benchmark results
reported in the thesis (those come from the evaluation scripts, which use
labeled test images).

Changes from the previous version (see README for the reasons):
  1. YOLO26n is run with its NMS-free one-to-one head explicitly
     (nms=False); YOLO11n uses its normal NMS head.
  2. Timing uses Ultralytics' own preprocess / inference / postprocess
     times instead of a wall-clock timer around predict().
  3. Each model is warmed up on blank frames before the camera starts.
  4. In "Compare both" mode the model order alternates every frame, so
     neither model always runs first.
  5. The confidence threshold and inference head are logged with every
     record.
  6. Only the sticker class is counted. If a checkpoint also has a
     privacy-blur class, those regions are pixelated on screen and are
     never counted as stickers.
  7. Comparison measurements are written to the JSONL log as they happen,
     so a crash does not lose the session.
  8. Field Test tab (Chapter 5): each capture saves the camera frame, runs
     both models on that same frame at a fixed threshold, and logs the
     test conditions (distance, lighting, motion, occlusion) with the
     results to research_outputs/field_test/field_log.csv.
  9. The webcam is read in its own thread and only the newest frame is
     used, so the picture does not fall behind real life.
"""

import csv
import hashlib
import json
import os
import queue
import threading
import time
import tkinter as tk

from datetime import datetime
from pathlib import Path
from tkinter import ttk, messagebox, filedialog

import cv2
import numpy as np
import torch
import pandas as pd
from PIL import Image, ImageTk
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "research_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Final training runs (70/15/15 split, 980/210/210 images).
# If you retrain with train.py, change these to
# runs/benchmark/yolo26n_s0/weights/best.pt and yolo11n_s0.
FINAL_RUNS_DIR = BASE_DIR / "runs" / "detect" / "carl-andre-de-castro" / "runs"

MODEL_PATHS = {
    "YOLO26n": FINAL_RUNS_DIR / "yolo26n_final_70_15_15" / "weights" / "best.pt",
    "YOLO11n": FINAL_RUNS_DIR / "yolo11n_final_70_15_15-2" / "weights" / "best.pt",
}

# Inference head per model (thesis Chapter IV, Step 8).
# YOLO26n: one-to-one, NMS-free head, selected with nms=False
# (Ultralytics 8.4.142+; older versions used end2end=True).
# YOLO11n: has only the NMS head, so no extra argument is passed.
INFERENCE_SETTINGS = {
    "YOLO26n": {"nms": False},
    "YOLO11n": {},
}
HEAD_LABELS = {
    "YOLO26n": "one-to-one (NMS-free)",
    "YOLO11n": "NMS",
}

# Class handling. A class whose name contains STICKER_KEYWORD is counted as
# a sticker. A class whose name contains BLUR_KEYWORD is treated as a
# privacy region (pixelated, never counted). If a model has exactly one
# class, that class is treated as the sticker.
STICKER_KEYWORD = "sticker"
BLUR_KEYWORD = "blur"
PRIVACY_BLUR = True
PIXEL_BLOCK = 12

HISTORY_FILE = OUTPUT_DIR / "live_detection_history.csv"
SESSION_FILE = OUTPUT_DIR / "live_detection_sessions.jsonl"

HISTORY_COLUMNS = [
    "timestamp",
    "session_id",
    "model",
    "inference_head",
    "class_name",
    "confidence",
    "conf_threshold",
    "preprocess_ms",
    "model_ms",
    "postprocess_ms",
    "inference_ms",
    "frame_count",
    "checkpoint_path",
    "checkpoint_sha256",
    "device",
]

CAMERA_INDEX = 0
IMAGE_SIZE = 640
# 640 x 480 keeps the live view smooth. The models shrink every frame to
# 640 pixels wide before detecting anyway, so a larger camera resolution
# does not change what the models see; it only makes the app slower.
# (The Silvertec SEW-001 can do 1920 x 1080 if bigger saved pictures are
# ever needed: change both numbers.) The resolution actually delivered is
# shown in the Field Test tab and saved with every capture.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ------------------------------------------------------------
# FIELD TEST (Chapter 5)
# ------------------------------------------------------------
FIELD_DIR = OUTPUT_DIR / "field_test"
FIELD_IMAGE_DIR = FIELD_DIR / "images"          # frames to label later
FIELD_ANNOTATED_DIR = FIELD_DIR / "annotated"   # both models' boxes, for figures
FIELD_DISCARD_DIR = FIELD_DIR / "_discarded"    # captures removed with Undo
FIELD_LOG = FIELD_DIR / "field_log.csv"

# Fixed threshold for every field capture (the live slider does not change it),
# so all captures are judged the same way.
FIELD_CONF = 0.25

# Condition choices shown in the Field Test tab. Edit to match the thesis.
FIELD_CONDITIONS = {
    "distance_m": ["6", "7", "8", "9", "10"],
    "lighting": ["morning", "noon", "afternoon", "dusk", "night"],
    "motion": ["parked", "slow", "normal speed"],
    "occlusion": ["none", "partial", "heavy"],
    "sticker_present": ["yes", "no"],
}
FIELD_CONDITION_LABELS = {
    "distance_m": "Distance (m)",
    "lighting": "Lighting",
    "motion": "Motion",
    "occlusion": "Occlusion",
    "sticker_present": "Sticker present",
}

FIELD_MODELS = ("YOLO26n", "YOLO11n")
FIELD_COLUMNS = (
    ["capture_id", "timestamp", "session_id", "image_file",
     "annotated_file"]
    + list(FIELD_CONDITIONS)
    + ["lux", "notes", "frame_width", "frame_height", "conf_threshold",
       "privacy_regions_blurred"]
    + [
        f"{model}_{field}"
        for model in FIELD_MODELS
        for field in ("detected", "count", "max_conf", "boxes_xyxy",
                      "inference_ms", "run_position")
    ]
)

WARMUP_FRAMES = 10          # thesis: 10 warm-up images before timing
LOG_INTERVAL_S = 1.0        # single-model history: at most one save per second

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def timestamp_now():
    return datetime.now().isoformat(timespec="seconds")


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def append_jsonl(record):
    """Append one JSON record without changing previous records."""
    with SESSION_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_history():
    """
    Read history safely.
    Rows with the wrong number of fields are skipped.
    The original CSV is never deleted or rewritten here.
    """
    records = []

    if not HISTORY_FILE.exists() or HISTORY_FILE.stat().st_size == 0:
        return records, 0

    skipped = 0

    try:
        with HISTORY_FILE.open(
            "r", newline="", encoding="utf-8-sig"
        ) as file:
            reader = csv.reader(file)
            header = next(reader, None)

            if not header:
                return [], 0

            header = [str(name).strip() for name in header]

            for row in reader:
                if not row or not any(str(value).strip() for value in row):
                    continue

                if len(row) != len(header):
                    skipped += 1
                    continue

                record = dict(zip(header, row))

                if not record.get("timestamp"):
                    skipped += 1
                    continue

                records.append(record)

    except (OSError, csv.Error, UnicodeError) as exc:
        raise RuntimeError(f"Could not read history CSV: {exc}") from exc

    return records, skipped


def _backup_history_file():
    backup = OUTPUT_DIR / (
        "live_detection_history_backup_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".csv"
    )
    os.replace(HISTORY_FILE, backup)


def rotate_old_history():
    """
    If the history file was written by an older app version (different
    columns), keep it as a timestamped backup and start a fresh file.
    """
    if not HISTORY_FILE.exists() or HISTORY_FILE.stat().st_size == 0:
        return False

    try:
        with HISTORY_FILE.open("r", newline="", encoding="utf-8-sig") as file:
            header = next(csv.reader(file), [])
    except (OSError, csv.Error, UnicodeError):
        header = None

    if header != HISTORY_COLUMNS:
        _backup_history_file()
        return True

    return False


def append_history(record):
    """
    Always write exactly the columns in HISTORY_COLUMNS.
    If an older history file has a different header, it is kept as a
    timestamped backup and a new file is started.
    """
    if HISTORY_FILE.exists() and HISTORY_FILE.stat().st_size > 0:
        try:
            with HISTORY_FILE.open(
                "r", newline="", encoding="utf-8-sig"
            ) as file:
                existing_header = next(csv.reader(file), [])

            if existing_header != HISTORY_COLUMNS:
                _backup_history_file()

        except (OSError, csv.Error, UnicodeError):
            _backup_history_file()

    new_file = (
        not HISTORY_FILE.exists()
        or HISTORY_FILE.stat().st_size == 0
    )

    with HISTORY_FILE.open(
        "a", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=HISTORY_COLUMNS,
            extrasaction="ignore",
        )

        if new_file:
            writer.writeheader()

        writer.writerow({
            column: record.get(column, "")
            for column in HISTORY_COLUMNS
        })


def read_field_log():
    """Return all field capture rows (list of dicts)."""
    if not FIELD_LOG.exists() or FIELD_LOG.stat().st_size == 0:
        return []

    with FIELD_LOG.open("r", newline="", encoding="utf-8-sig") as file:
        return [row for row in csv.DictReader(file) if row.get("capture_id")]


def write_field_rows(rows):
    """Rewrite the whole field log (used only by Undo)."""
    FIELD_DIR.mkdir(parents=True, exist_ok=True)
    temporary = FIELD_LOG.with_suffix(".tmp")

    with temporary.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELD_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temporary, FIELD_LOG)


def append_field_row(row):
    """Append one capture. An old-format log is kept as a backup first."""
    FIELD_DIR.mkdir(parents=True, exist_ok=True)

    if FIELD_LOG.exists() and FIELD_LOG.stat().st_size > 0:
        with FIELD_LOG.open("r", newline="", encoding="utf-8-sig") as file:
            header = next(csv.reader(file), [])

        if header != FIELD_COLUMNS:
            os.replace(FIELD_LOG, FIELD_DIR / (
                "field_log_backup_"
                + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
            ))

    new_file = not FIELD_LOG.exists() or FIELD_LOG.stat().st_size == 0

    with FIELD_LOG.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELD_COLUMNS,
                                extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _level_sort_key(value):
    try:
        return (0, float(value), "")
    except (TypeError, ValueError):
        order = {v: i for values in FIELD_CONDITIONS.values()
                 for i, v in enumerate(values)}
        return (1, order.get(value, 999), str(value))


def summarize_field(rows, factor):
    """
    Group field captures by one condition and return one summary per level.

    Captures WITH a sticker give the detection rate (frames in which the
    model found at least one sticker). Captures WITHOUT a sticker give the
    false-alarm rate. These are frame-level webcam results; precision,
    recall and mAP need the saved frames to be labelled first.
    """
    levels = {}

    for row in rows:
        levels.setdefault(row.get(factor, ""), []).append(row)

    summary = []

    for level in sorted(levels, key=_level_sort_key):
        group = levels[level]
        with_sticker = [r for r in group if r.get("sticker_present") == "yes"]
        without = [r for r in group if r.get("sticker_present") == "no"]
        item = {
            "factor": factor,
            "level": level,
            "captures": len(group),
            "with_sticker": len(with_sticker),
            "without_sticker": len(without),
        }

        for model in FIELD_MODELS:
            hits = [r for r in with_sticker if r.get(f"{model}_detected") == "1"]
            false_alarms = [r for r in without
                            if r.get(f"{model}_detected") == "1"]
            confidences = [float(r[f"{model}_max_conf"]) for r in hits
                           if r.get(f"{model}_max_conf")]

            item[f"{model}_detected"] = len(hits)
            item[f"{model}_detection_rate"] = (
                len(hits) / len(with_sticker) if with_sticker else None
            )
            item[f"{model}_avg_conf"] = (
                sum(confidences) / len(confidences) if confidences else None
            )
            item[f"{model}_false_alarms"] = len(false_alarms)
            item[f"{model}_false_alarm_rate"] = (
                len(false_alarms) / len(without) if without else None
            )

        summary.append(item)

    return summary


def encode_image(path, image, quality=95):
    """cv2.imwrite fails on some Windows paths; encode then write bytes."""
    ok, buffer = cv2.imencode(".jpg", image,
                              [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Could not encode image {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(buffer.tobytes())


class LatestFrameGrabber:
    """
    Reads the webcam in its own thread and keeps only the newest frame.

    Without this, frames queue up inside the camera driver while the models
    are busy, and the picture on screen falls further and further behind
    real life. Old frames are simply dropped.
    """

    def __init__(self, camera):
        self.camera = camera
        self.frame = None
        self.frame_id = 0
        self.failed = False
        self.condition = threading.Condition()
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        while not self.stopped:
            ok, frame = self.camera.read()

            with self.condition:
                if not ok:
                    self.failed = True
                    self.condition.notify_all()
                    return
                self.frame = frame
                self.frame_id += 1
                self.condition.notify_all()

    def read_new(self, last_id, timeout=3.0):
        """Wait for a frame newer than last_id. Returns (frame, id)."""
        with self.condition:
            self.condition.wait_for(
                lambda: self.frame_id != last_id or self.failed
                or self.stopped,
                timeout=timeout,
            )
            if self.frame_id == last_id:
                return None, last_id
            return self.frame, self.frame_id

    def stop(self):
        self.stopped = True
        with self.condition:
            self.condition.notify_all()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)


def classify_names(names):
    """
    Return (sticker_ids, blur_ids) for a model's class-name dictionary.
    """
    sticker_ids, blur_ids = set(), set()

    for class_id, class_name in names.items():
        lowered = str(class_name).lower()

        if BLUR_KEYWORD in lowered:
            blur_ids.add(int(class_id))
        elif STICKER_KEYWORD in lowered:
            sticker_ids.add(int(class_id))

    if not sticker_ids and len(names) == 1:
        sticker_ids = {int(next(iter(names)))}

    return sticker_ids, blur_ids


def split_detections(result, sticker_ids, blur_ids):
    """
    Split a result into sticker detections and privacy regions.
    Each sticker is (x1, y1, x2, y2, confidence, class_name).
    Each privacy region is (x1, y1, x2, y2).
    """
    stickers, privacy = [], []
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return stickers, privacy

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, classes):
        box = (int(x1), int(y1), int(x2), int(y2))

        if class_id in blur_ids:
            privacy.append(box)
        elif class_id in sticker_ids:
            stickers.append((
                *box,
                float(conf),
                str(result.names.get(class_id, class_id)),
            ))

    return stickers, privacy


def pixelate_regions(image, regions):
    height, width = image.shape[:2]

    for x1, y1, x2, y2 in regions:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        roi = image[y1:y2, x1:x2]
        small = cv2.resize(
            roi,
            (
                max(1, (x2 - x1) // PIXEL_BLOCK),
                max(1, (y2 - y1) // PIXEL_BLOCK),
            ),
            interpolation=cv2.INTER_LINEAR,
        )
        image[y1:y2, x1:x2] = cv2.resize(
            small,
            (x2 - x1, y2 - y1),
            interpolation=cv2.INTER_NEAREST,
        )

    return image


def draw_frame(frame, stickers, privacy):
    """Pixelate privacy regions, then draw sticker boxes. Returns BGR."""
    image = frame.copy()

    if PRIVACY_BLUR and privacy:
        pixelate_regions(image, privacy)

    for x1, y1, x2, y2, conf, _name in stickers:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            image,
            f"sticker {conf:.2f}",
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )

    return image


def speed_breakdown(result):
    """Ultralytics' own per-image timings in ms."""
    speed = getattr(result, "speed", {}) or {}
    pre = float(speed.get("preprocess") or 0.0)
    model_ms = float(speed.get("inference") or 0.0)
    post = float(speed.get("postprocess") or 0.0)
    return pre, model_ms, post, pre + model_ms + post


def model_evidence(name, model, checkpoint, head_label):
    network = getattr(model, "model", None)
    yaml_data = getattr(network, "yaml", {}) or {}

    try:
        parameter_count = sum(
            parameter.numel()
            for parameter in network.parameters()
        )
    except Exception:
        parameter_count = None

    return {
        "model_name": name,
        "checkpoint_path": str(checkpoint.resolve()),
        "sha256": sha256_file(checkpoint),
        "task": getattr(model, "task", "unknown"),
        "architecture_scale": yaml_data.get("scale", ""),
        "parameter_count": parameter_count,
        "class_names": model.names,
        "inference_head": head_label,
        "inference_settings": INFERENCE_SETTINGS.get(name, {}),
        "ultralytics_version": __import__("ultralytics").__version__,
        "pytorch_version": torch.__version__,
        "device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
        "verified_at": timestamp_now(),
    }


# ============================================================
# MAIN APPLICATION
# ============================================================

class StickerDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PSU Vehicle Sticker Detection — Model Comparison")
        self.root.geometry("1200x780")
        self.root.minsize(920, 620)

        self.camera = None
        self.running = False
        self.closing = False
        self.worker = None

        self.models = {}
        self.evidence = {}
        self.class_ids = {}          # name -> (sticker_ids, blur_ids)
        self.predict_args = {}       # name -> extra predict() arguments
        self.head_labels = {}        # name -> head actually used
        self.model_name = None
        self.session_id = None
        self.photo = None
        self.frame_index = 0

        self.current_conf = 0.25
        self.lock = threading.Lock()
        self.last_saved_time = 0.0

        self.comparison_samples = {
            name: [] for name in MODEL_PATHS
        }

        # Field test state
        self.capture_queue = queue.Queue()
        self.field_capture_count = 0
        self.actual_resolution = (0, 0)
        self.field_photo = None
        self.field_vars = {
            key: tk.StringVar(value=values[0])
            for key, values in FIELD_CONDITIONS.items()
        }
        self.field_lux_var = tk.StringVar()
        self.field_notes_var = tk.StringVar()
        self.field_auto_var = tk.BooleanVar(value=False)
        self.field_interval_var = tk.IntVar(value=3)
        self.field_group_var = tk.StringVar(value="distance_m")

        self.model_var = tk.StringVar(value="Compare both")
        self.conf_var = tk.DoubleVar(value=0.25)
        self.status_var = tk.StringVar(
            value="Ready. Select a model and start the camera."
        )

        self._build_interface()

        if rotate_old_history():
            self.status_var.set(
                "Old-format detection history saved as a backup in "
                "research_outputs. A new history file has been started."
            )

        self.refresh_history()
        self.refresh_field_summary()

        self.root.bind("<space>", self._space_pressed)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # --------------------------------------------------------
    # INTERFACE
    # --------------------------------------------------------

    def _build_interface(self):
        style = ttk.Style()

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "AppTitle.TLabel",
            font=("Segoe UI", 17, "bold"),
        )

        header = ttk.Frame(self.root, padding=12)
        header.pack(fill="x")

        ttk.Label(
            header,
            text="VEHICLE STICKER DETECTION",
            style="AppTitle.TLabel",
        ).pack(side="left")

        ttk.Label(
            header,
            text="Research demonstrator • Not a deployed system",
        ).pack(side="right")

        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=5)

        self.live_tab = ttk.Frame(self.tabs, padding=10)
        self.compare_tab = ttk.Frame(self.tabs, padding=10)
        self.history_tab = ttk.Frame(self.tabs, padding=10)
        self.field_tab = ttk.Frame(self.tabs, padding=10)

        self.tabs.add(self.live_tab, text="Live Detection")
        self.tabs.add(self.compare_tab, text="Model Comparison")
        self.tabs.add(self.field_tab, text="Field Test")
        self.tabs.add(self.history_tab, text="History & Model Evidence")

        self._build_live_tab()
        self._build_compare_tab()
        self._build_field_tab()
        self._build_history_tab()

        # Packed at the bottom before the tabs, so it is always visible.
        ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w",
            padding=5,
        ).pack(fill="x", side="bottom", before=self.tabs)

    def _build_live_tab(self):
        controls = ttk.LabelFrame(
            self.live_tab,
            text="Controls",
            padding=8,
        )
        controls.pack(fill="x", pady=(0, 8))

        ttk.Label(controls, text="Model:").grid(
            row=0, column=0, padx=4, pady=4
        )

        self.model_selector = ttk.Combobox(
            controls,
            textvariable=self.model_var,
            values=["Compare both", "YOLO26n", "YOLO11n"],
            state="readonly",
            width=15,
        )
        self.model_selector.grid(row=0, column=1, padx=4)
        self.model_selector.bind(
            "<<ComboboxSelected>>",
            self.selection_changed,
        )

        ttk.Label(
            controls,
            text="Confidence threshold:",
        ).grid(row=0, column=2, padx=(12, 4))

        ttk.Scale(
            controls,
            from_=0.10,
            to=0.90,
            variable=self.conf_var,
            orient="horizontal",
            length=130,
            command=self.confidence_changed,
        ).grid(row=0, column=3, padx=4)

        self.conf_label = ttk.Label(controls, text="0.25")
        self.conf_label.grid(row=0, column=4, padx=4)

        self.start_button = ttk.Button(
            controls,
            text="Start Camera",
            command=self.start_camera,
        )
        self.start_button.grid(row=0, column=5, padx=(12, 4))

        self.stop_button = ttk.Button(
            controls,
            text="Stop Camera",
            command=self.stop_camera,
            state="disabled",
        )
        self.stop_button.grid(row=0, column=6, padx=4)

        content = ttk.Frame(self.live_tab)
        content.pack(fill="both", expand=True)

        video_frame = ttk.LabelFrame(
            content,
            text="Camera feed",
            padding=6,
        )
        video_frame.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 8),
        )

        self.video_label = ttk.Label(
            video_frame,
            text="Camera stopped\nClick Start Camera",
            anchor="center",
            justify="center",
        )
        self.video_label.pack(fill="both", expand=True)

        stats_frame = ttk.LabelFrame(
            content,
            text="Live statistics",
            padding=10,
            width=270,
        )
        stats_frame.pack(side="right", fill="y")
        stats_frame.pack_propagate(False)

        self.stat_labels = {}

        stat_items = [
            ("Selected model", "model"),
            ("Inference head", "head"),
            ("Checkpoint SHA-256", "hash"),
            ("Stickers in current frame", "count"),
            ("Time per image (ms): pre / model / post", "inference"),
            ("Approx. FPS (end to end)", "fps"),
            ("Device", "device"),
        ]

        for title, key in stat_items:
            ttk.Label(stats_frame, text=title, wraplength=240).pack(
                anchor="w",
                pady=(7, 1),
            )

            label = ttk.Label(
                stats_frame,
                text="—",
                wraplength=240,
                font=("Segoe UI", 10, "bold"),
            )
            label.pack(anchor="w")
            self.stat_labels[key] = label

        self.stat_labels["device"].configure(
            text=(
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "CPU"
            )
        )

        ttk.Button(
            stats_frame,
            text="Open history",
            command=lambda: self.tabs.select(self.history_tab),
        ).pack(fill="x", pady=(15, 5))

        ttk.Label(
            stats_frame,
            text=(
                "Webcam measurements show live behaviour. "
                "They do not establish accuracy, precision, recall, "
                "or mAP without ground-truth labels."
            ),
            wraplength=240,
        ).pack(anchor="w", pady=8)

    def _build_compare_tab(self):
        ttk.Label(
            self.compare_tab,
            text="YOLO26n vs YOLO11n",
            style="AppTitle.TLabel",
        ).pack(anchor="w", pady=(0, 5))

        ttk.Label(
            self.compare_tab,
            text=(
                "Choose Compare both on the Live Detection tab. "
                "Both models process the same captured frames, in "
                "alternating order. Stop the camera to view the summary."
            ),
            wraplength=1050,
        ).pack(anchor="w", pady=(0, 10))

        self.comparison_heading = ttk.Label(
            self.compare_tab,
            text="No comparison session yet.",
            font=("Segoe UI", 10, "bold"),
        )
        self.comparison_heading.pack(anchor="w", pady=(0, 8))

        columns = (
            "model",
            "head",
            "frames",
            "detected_frames",
            "avg_objects",
            "avg_confidence",
            "avg_model_ms",
            "avg_ms",
            "fps",
        )

        self.comparison_table = ttk.Treeview(
            self.compare_tab,
            columns=columns,
            show="headings",
            height=4,
        )

        headings = {
            "model": ("Model", 90),
            "head": ("Head", 140),
            "frames": ("Frames", 70),
            "detected_frames": ("Frames with sticker", 125),
            "avg_objects": ("Avg stickers/frame", 120),
            "avg_confidence": ("Avg confidence", 105),
            "avg_model_ms": ("Avg model (ms)", 105),
            "avg_ms": ("Avg end-to-end (ms)", 130),
            "fps": ("Approx. FPS", 90),
        }

        for key, (label, width) in headings.items():
            self.comparison_table.heading(key, text=label)
            self.comparison_table.column(
                key,
                width=width,
                anchor="center",
            )

        self.comparison_table.pack(fill="x", pady=(0, 12))

        chart_frame = ttk.LabelFrame(
            self.compare_tab,
            text="Average end-to-end time per image (lower is faster)",
            padding=8,
        )
        chart_frame.pack(fill="x", pady=(0, 10))

        self.chart = tk.Canvas(
            chart_frame,
            height=120,
            background="white",
            highlightthickness=0,
        )
        self.chart.pack(fill="x", expand=True)

        self.comparison_summary = ttk.Label(
            self.compare_tab,
            text="The comparison summary will appear here.",
            wraplength=1050,
            justify="left",
        )
        self.comparison_summary.pack(anchor="w", pady=8)

        ttk.Button(
            self.compare_tab,
            text="Export comparison CSV",
            command=self.export_comparison,
        ).pack(anchor="e", pady=5)

    # --------------------------------------------------------
    # FIELD TEST TAB (Chapter 5)
    # --------------------------------------------------------

    def _build_field_tab(self):
        top = ttk.Frame(self.field_tab)
        top.pack(fill="x")

        ttk.Label(
            top,
            text=(
                "1. Start the camera (both models).  2. Set the conditions "
                "for the next shot.  3. Press Capture or the Space bar. "
                f"Every capture uses a fixed threshold of {FIELD_CONF:.2f}."
            ),
            wraplength=1100,
        ).pack(anchor="w", pady=(0, 6))

        conditions = ttk.LabelFrame(top, text="Conditions for the next "
                                    "capture", padding=8)
        conditions.pack(fill="x")

        for column, (key, values) in enumerate(FIELD_CONDITIONS.items()):
            ttk.Label(conditions, text=FIELD_CONDITION_LABELS[key]).grid(
                row=0, column=column, padx=6, sticky="w"
            )
            ttk.Combobox(
                conditions,
                textvariable=self.field_vars[key],
                values=values,
                state="readonly",
                width=13,
            ).grid(row=1, column=column, padx=6, pady=(0, 4))

        extra = len(FIELD_CONDITIONS)
        ttk.Label(conditions, text="Lux (optional)").grid(
            row=0, column=extra, padx=6, sticky="w")
        ttk.Entry(conditions, textvariable=self.field_lux_var,
                  width=9).grid(row=1, column=extra, padx=6, pady=(0, 4))
        ttk.Label(conditions, text="Notes (optional)").grid(
            row=0, column=extra + 1, padx=6, sticky="w")
        ttk.Entry(conditions, textvariable=self.field_notes_var,
                  width=26).grid(row=1, column=extra + 1, padx=6,
                                 pady=(0, 4))

        actions = ttk.Frame(top)
        actions.pack(fill="x", pady=8)

        self.field_start_button = ttk.Button(
            actions, text="Start camera (both models)",
            command=self.start_field_camera)
        self.field_start_button.pack(side="left", padx=(0, 4))

        self.field_stop_button = ttk.Button(
            actions, text="Stop camera", command=self.stop_camera,
            state="disabled")
        self.field_stop_button.pack(side="left", padx=4)

        self.capture_button = ttk.Button(
            actions, text="Capture  (Space)", command=self.request_capture,
            state="disabled")
        self.capture_button.pack(side="left", padx=(16, 4))

        ttk.Checkbutton(
            actions, text="Auto-capture every",
            variable=self.field_auto_var,
            command=self._auto_capture_toggled,
        ).pack(side="left", padx=(16, 2))
        ttk.Spinbox(actions, from_=1, to=30, width=4,
                    textvariable=self.field_interval_var).pack(side="left")
        ttk.Label(actions, text="s").pack(side="left", padx=(2, 16))

        ttk.Button(actions, text="Undo last capture",
                   command=self.undo_last_capture).pack(side="left", padx=4)
        ttk.Button(actions, text="Open field folder",
                   command=self.open_field_folder).pack(side="left", padx=4)

        middle = ttk.Frame(self.field_tab)
        middle.pack(fill="both", expand=True)

        preview = ttk.LabelFrame(middle, text="Live view (YOLO26n left, "
                                 "YOLO11n right)", padding=4)
        preview.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self.field_live_label = ttk.Label(
            preview, text="Camera stopped", anchor="center")
        self.field_live_label.pack(fill="both", expand=True)

        info = ttk.LabelFrame(middle, text="Last capture", padding=8,
                              width=300)
        info.pack(side="right", fill="y")
        info.pack_propagate(False)

        self.field_count_label = ttk.Label(
            info, text="Captures saved: 0", font=("Segoe UI", 10, "bold"))
        self.field_count_label.pack(anchor="w")

        self.field_resolution_label = ttk.Label(
            info, text="Camera resolution: —", wraplength=280)
        self.field_resolution_label.pack(anchor="w", pady=(4, 8))

        self.field_last_label = ttk.Label(
            info, text="No capture yet.", wraplength=280, justify="left")
        self.field_last_label.pack(anchor="w")

        bottom = ttk.LabelFrame(
            self.field_tab,
            text=("Quick results (frame level: detection rate on captures "
                  "with a sticker, false alarms on captures without one)"),
            padding=6,
        )
        bottom.pack(fill="x", pady=(8, 0))

        group_row = ttk.Frame(bottom)
        group_row.pack(fill="x", pady=(0, 4))
        ttk.Label(group_row, text="Group by:").pack(side="left")
        group_box = ttk.Combobox(
            group_row, textvariable=self.field_group_var,
            values=[k for k in FIELD_CONDITIONS if k != "sticker_present"],
            state="readonly", width=14)
        group_box.pack(side="left", padx=4)
        group_box.bind("<<ComboboxSelected>>",
                       lambda _e: self.refresh_field_summary())
        ttk.Button(group_row, text="Export summary CSV",
                   command=self.export_field_summary).pack(side="right")

        columns = ("level", "with", "d26", "d11", "c26", "c11",
                   "without", "f26", "f11")
        headings = {
            "level": ("Level", 90),
            "with": ("With sticker", 95),
            "d26": ("26n detected", 120),
            "d11": ("11n detected", 120),
            "c26": ("26n avg conf", 95),
            "c11": ("11n avg conf", 95),
            "without": ("No sticker", 90),
            "f26": ("26n false alarms", 125),
            "f11": ("11n false alarms", 125),
        }
        self.field_table = ttk.Treeview(bottom, columns=columns,
                                        show="headings", height=5)
        for key, (label, width) in headings.items():
            self.field_table.heading(key, text=label)
            self.field_table.column(key, width=width, anchor="center")
        self.field_table.pack(fill="x")

    def start_field_camera(self):
        self.model_var.set("Compare both")
        self.start_camera()

    def _field_conditions(self):
        return {key: var.get() for key, var in self.field_vars.items()}

    def _space_pressed(self, _event=None):
        # Space typed into a text box, or pressed on a focused button,
        # belongs to that widget (the Capture button captures by itself).
        focused = self.root.focus_get()
        if isinstance(focused, (ttk.Entry, ttk.Button, ttk.Checkbutton,
                                tk.Entry, tk.Button)):
            return None

        if self.tabs.select() == str(self.field_tab):
            self.request_capture()
            return "break"
        return None

    def request_capture(self):
        if not self.running or len(self.models) != 2:
            messagebox.showinfo(
                "Camera not ready",
                "Start the camera with both models first "
                "(Start camera button on this tab).",
            )
            return

        lux = self.field_lux_var.get().strip()
        if lux:
            try:
                float(lux)
            except ValueError:
                messagebox.showerror("Lux", "Lux must be a number or empty.")
                return

        self.capture_queue.put({
            **self._field_conditions(),
            "lux": lux,
            "notes": self.field_notes_var.get().strip(),
        })
        self.status_var.set("Capturing...")

    def _auto_capture_toggled(self):
        if self.field_auto_var.get():
            self._auto_capture_tick()

    def _auto_capture_tick(self):
        if not self.field_auto_var.get() or self.closing:
            return

        if self.running and len(self.models) == 2:
            if self.capture_queue.empty():
                self.request_capture()
        else:
            self.field_auto_var.set(False)
            return

        try:
            seconds = max(1, int(self.field_interval_var.get()))
        except (tk.TclError, ValueError):
            seconds = 3

        self.root.after(seconds * 1000, self._auto_capture_tick)

    def _process_field_capture(self, frame, request):
        """Runs in the camera thread on the exact frame that was read."""
        self.field_capture_count += 1
        now = datetime.now()
        capture_id = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        stem = f"field_{capture_id}_d{request['distance_m']}m"

        order = (FIELD_MODELS if self.field_capture_count % 2
                 else tuple(reversed(FIELD_MODELS)))

        results, privacy_regions = {}, []

        for position, name in enumerate(order, start=1):
            result = self._predict(name, frame.copy(), FIELD_CONF)
            sticker_ids, blur_ids = self.class_ids[name]
            stickers, privacy = split_detections(result, sticker_ids,
                                                 blur_ids)
            privacy_regions += privacy
            _pre, _m, _post, total_ms = speed_breakdown(result)
            results[name] = {
                "stickers": stickers,
                "privacy": privacy,
                "inference_ms": total_ms,
                "run_position": position,
            }

        # The saved frame is the one that will be labelled, so it has no
        # boxes drawn on it. Privacy regions are pixelated if found.
        clean = frame.copy()
        if PRIVACY_BLUR and privacy_regions:
            pixelate_regions(clean, privacy_regions)

        image_path = FIELD_IMAGE_DIR / f"{stem}.jpg"
        encode_image(image_path, clean)

        panels = []
        for name in FIELD_MODELS:
            panel = draw_frame(clean, results[name]["stickers"], [])
            cv2.putText(panel, f"{name}: {len(results[name]['stickers'])}"
                        " sticker(s)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.4, (0, 255, 255), 3, cv2.LINE_AA)
            panels.append(panel)
        annotated_path = FIELD_ANNOTATED_DIR / f"{stem}_both.jpg"
        encode_image(annotated_path, cv2.hconcat(panels), quality=85)

        height, width = frame.shape[:2]
        row = {
            "capture_id": capture_id,
            "timestamp": now.isoformat(timespec="seconds"),
            "session_id": self.session_id,
            "image_file": str(image_path.relative_to(FIELD_DIR)),
            "annotated_file": str(annotated_path.relative_to(FIELD_DIR)),
            **{key: request[key] for key in FIELD_CONDITIONS},
            "lux": request["lux"],
            "notes": request["notes"],
            "frame_width": width,
            "frame_height": height,
            "conf_threshold": FIELD_CONF,
            "privacy_regions_blurred": len(privacy_regions),
        }

        for name in FIELD_MODELS:
            stickers = results[name]["stickers"]
            row[f"{name}_detected"] = int(bool(stickers))
            row[f"{name}_count"] = len(stickers)
            row[f"{name}_max_conf"] = (
                round(max(s[4] for s in stickers), 5) if stickers else ""
            )
            row[f"{name}_boxes_xyxy"] = json.dumps(
                [[*s[:4], round(s[4], 5)] for s in stickers])
            row[f"{name}_inference_ms"] = round(
                results[name]["inference_ms"], 3)
            row[f"{name}_run_position"] = results[name]["run_position"]

        append_field_row(row)
        append_jsonl({"record_type": "field_capture", **row})

        self.root.after(0, lambda r=row: self._field_capture_saved(r))

    def _field_capture_saved(self, row):
        if self.closing:
            return

        def describe(name):
            if row[f"{name}_detected"]:
                return (f"{name}: FOUND {row[f'{name}_count']} "
                        f"(conf {float(row[f'{name}_max_conf']):.2f})")
            return f"{name}: not found"

        self.field_last_label.configure(text=(
            f"{row['capture_id']}\n"
            f"{row['distance_m']} m | {row['lighting']} | {row['motion']} | "
            f"occlusion {row['occlusion']} | sticker {row['sticker_present']}"
            f"\n\n{describe('YOLO26n')}\n{describe('YOLO11n')}"
        ))
        self.status_var.set(f"Capture saved: {row['image_file']}")
        self.refresh_field_summary()

    def refresh_field_summary(self):
        if not hasattr(self, "field_table"):
            return

        try:
            rows = read_field_log()
        except Exception as exc:
            self.status_var.set(f"Could not read field log: {exc}")
            return

        self.field_count_label.configure(text=f"Captures saved: {len(rows)}")

        for item in self.field_table.get_children():
            self.field_table.delete(item)

        def rate(hits, total, value):
            return f"{hits}/{total} ({value * 100:.0f}%)" if total else "—"

        def conf(value):
            return f"{value:.2f}" if value is not None else "—"

        for item in summarize_field(rows, self.field_group_var.get()):
            self.field_table.insert("", "end", values=(
                item["level"],
                item["with_sticker"],
                rate(item["YOLO26n_detected"], item["with_sticker"],
                     item["YOLO26n_detection_rate"] or 0),
                rate(item["YOLO11n_detected"], item["with_sticker"],
                     item["YOLO11n_detection_rate"] or 0),
                conf(item["YOLO26n_avg_conf"]),
                conf(item["YOLO11n_avg_conf"]),
                item["without_sticker"],
                rate(item["YOLO26n_false_alarms"], item["without_sticker"],
                     item["YOLO26n_false_alarm_rate"] or 0),
                rate(item["YOLO11n_false_alarms"], item["without_sticker"],
                     item["YOLO11n_false_alarm_rate"] or 0),
            ))

    def export_field_summary(self):
        rows = read_field_log()
        if not rows:
            messagebox.showinfo("No captures", "No field captures yet.")
            return

        summary = []
        for factor in FIELD_CONDITIONS:
            if factor != "sticker_present":
                summary += summarize_field(rows, factor)

        destination = FIELD_DIR / (
            "field_summary_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".csv")
        pd.DataFrame(summary).to_csv(destination, index=False)
        messagebox.showinfo("Export complete",
                            f"Summary for every condition saved to:\n"
                            f"{destination}")

    def undo_last_capture(self):
        rows = read_field_log()
        if not rows:
            messagebox.showinfo("Nothing to undo", "No field captures yet.")
            return

        last = rows[-1]
        if not messagebox.askyesno(
            "Undo last capture",
            f"Remove capture {last['capture_id']} "
            f"({last['distance_m']} m, {last['lighting']})?\n\n"
            "Its images are moved to the _discarded folder, not deleted.",
        ):
            return

        FIELD_DISCARD_DIR.mkdir(parents=True, exist_ok=True)
        for key in ("image_file", "annotated_file"):
            source = FIELD_DIR / last.get(key, "")
            if last.get(key) and source.is_file():
                os.replace(source, FIELD_DISCARD_DIR / source.name)

        write_field_rows(rows[:-1])
        append_jsonl({"record_type": "field_capture_undone",
                      "capture_id": last["capture_id"],
                      "timestamp": timestamp_now()})
        self.field_last_label.configure(
            text=f"Capture {last['capture_id']} removed.")
        self.refresh_field_summary()

    def open_field_folder(self):
        FIELD_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(FIELD_DIR)          # Windows
        except AttributeError:
            messagebox.showinfo("Field folder", str(FIELD_DIR))

    def _build_history_tab(self):
        buttons = ttk.Frame(self.history_tab)
        buttons.pack(fill="x", pady=(0, 8))

        self.history_count_label = ttk.Label(
            buttons,
            text="Saved detections: 0",
            font=("Segoe UI", 10, "bold"),
        )
        self.history_count_label.pack(side="left")

        ttk.Button(
            buttons,
            text="Refresh",
            command=self.refresh_history,
        ).pack(side="right", padx=3)

        ttk.Button(
            buttons,
            text="Export history CSV",
            command=self.export_history,
        ).pack(side="right", padx=3)

        ttk.Button(
            buttons,
            text="Export model evidence JSON",
            command=self.export_evidence,
        ).pack(side="right", padx=3)

        columns = tuple(HISTORY_COLUMNS)

        self.history_table = ttk.Treeview(
            self.history_tab,
            columns=columns,
            show="headings",
        )

        widths = {
            "timestamp": 145,
            "session_id": 150,
            "model": 85,
            "inference_head": 140,
            "class_name": 110,
            "confidence": 90,
            "conf_threshold": 100,
            "preprocess_ms": 95,
            "model_ms": 85,
            "postprocess_ms": 100,
            "inference_ms": 95,
            "frame_count": 90,
            "checkpoint_path": 230,
            "checkpoint_sha256": 240,
            "device": 130,
        }

        for key in columns:
            self.history_table.heading(
                key, text=key.replace("_", " ").title()
            )
            self.history_table.column(
                key,
                width=widths.get(key, 110),
                anchor="center",
            )

        scrollbar_y = ttk.Scrollbar(
            self.history_tab,
            orient="vertical",
            command=self.history_table.yview,
        )
        scrollbar_x = ttk.Scrollbar(
            self.history_tab,
            orient="horizontal",
            command=self.history_table.xview,
        )

        self.history_table.configure(
            yscrollcommand=scrollbar_y.set,
            xscrollcommand=scrollbar_x.set,
        )

        self.history_table.pack(
            side="top",
            fill="both",
            expand=True,
        )
        scrollbar_y.pack(side="right", fill="y")
        scrollbar_x.pack(side="bottom", fill="x")

    # --------------------------------------------------------
    # MODEL AND SETTINGS
    # --------------------------------------------------------

    def confidence_changed(self, value):
        self.current_conf = float(value)
        self.conf_label.configure(text=f"{self.current_conf:.2f}")

    def selection_changed(self, _event=None):
        if self.running:
            messagebox.showinfo(
                "Camera running",
                "Stop the camera before changing the model.",
            )
            return

        self.models = {}
        self.evidence = {}

        self.stat_labels["model"].configure(text="—")
        self.stat_labels["head"].configure(text="—")
        self.stat_labels["hash"].configure(text="—")
        self.status_var.set(
            "Selection updated. Start the camera to load the model."
        )

    def _predict(self, name, frame, conf):
        """Run one model with its configured inference head."""
        return self.models[name].predict(
            source=frame,
            conf=conf,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
            **self.predict_args[name],
        )[0]

    def _load_model(self, name):
        checkpoint = MODEL_PATHS[name]

        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"{name} checkpoint was not found:\n\n{checkpoint}\n\n"
                "Check that the best.pt file exists at this path."
            )

        model = YOLO(str(checkpoint))
        self.models[name] = model
        self.class_ids[name] = classify_names(model.names)

        if not self.class_ids[name][0]:
            raise ValueError(
                f"{name}: no class name contains '{STICKER_KEYWORD}'. "
                f"Classes found: {model.names}. Update STICKER_KEYWORD."
            )

        # Warm up and confirm the requested inference head is supported.
        self.predict_args[name] = dict(INFERENCE_SETTINGS.get(name, {}))
        self.head_labels[name] = HEAD_LABELS.get(name, "default")
        blank = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

        try:
            self._predict(name, blank, 0.25)
        except Exception as exc:
            if not self.predict_args[name]:
                raise
            raise RuntimeError(
                f"{name}: this Ultralytics version rejected "
                f"{self.predict_args[name]}.\n\n{exc}\n\n"
                "Update Ultralytics (pip install -U ultralytics) or set "
                "the matching option in INFERENCE_SETTINGS."
            ) from exc

        for _ in range(WARMUP_FRAMES - 1):
            self._predict(name, blank, 0.25)

        evidence = model_evidence(
            name, model, checkpoint, self.head_labels[name]
        )
        self.evidence[name] = evidence

        append_jsonl({
            "record_type": "model_loaded",
            "warmup_frames": WARMUP_FRAMES,
            **evidence,
        })

    # --------------------------------------------------------
    # CAMERA START / STOP
    # --------------------------------------------------------

    def start_camera(self):
        if self.running:
            return

        try:
            self.models = {}
            self.evidence = {}
            self.class_ids = {}
            self.predict_args = {}
            self.head_labels = {}

            selection = self.model_var.get()

            names = (
                list(MODEL_PATHS.keys())
                if selection == "Compare both"
                else [selection]
            )

            for name in names:
                self.status_var.set(f"Loading and warming up {name}...")
                self.root.update_idletasks()
                self._load_model(name)

            self.model_name = selection

        except Exception as exc:
            messagebox.showerror("Model loading error", str(exc))
            self.status_var.set("Model loading failed.")
            return

        camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

        if not camera.isOpened():
            camera.release()
            messagebox.showerror(
                "Camera error",
                "Could not open webcam 0. Close other applications "
                "using the camera and try again.",
            )
            self.status_var.set("Camera could not be opened.")
            return

        # MJPG lets most USB webcams deliver higher resolutions at full speed.
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        ok, first_frame = camera.read()
        if ok:
            self.actual_resolution = (first_frame.shape[1],
                                      first_frame.shape[0])
        else:
            self.actual_resolution = (
                int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )

        width, height = self.actual_resolution
        resolution_text = f"Camera resolution: {width} x {height}"
        if (width, height) != (CAMERA_WIDTH, CAMERA_HEIGHT):
            resolution_text += (f"  (requested {CAMERA_WIDTH} x "
                                f"{CAMERA_HEIGHT})")
        self.field_resolution_label.configure(text=resolution_text)

        while not self.capture_queue.empty():   # drop stale requests
            self.capture_queue.get_nowait()

        self.camera = camera
        self.running = True
        self.last_saved_time = 0.0
        self.frame_index = 0

        self.session_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_"
            + selection.replace(" ", "_")
        )

        self.comparison_samples = {
            name: [] for name in names
        }

        append_jsonl({
            "record_type": "session_start",
            "session_id": self.session_id,
            "selection": selection,
            "models": names,
            "inference_heads": dict(self.head_labels),
            "conf_threshold_at_start": round(self.current_conf, 3),
            "image_size": IMAGE_SIZE,
            "camera_resolution": list(self.actual_resolution),
            "privacy_blur": PRIVACY_BLUR,
            "started_at": timestamp_now(),
        })

        if selection == "Compare both":
            self.stat_labels["model"].configure(text="Compare both")
            self.stat_labels["head"].configure(
                text=" | ".join(
                    f"{n}: {self.head_labels[n]}" for n in names
                )
            )
            self.stat_labels["hash"].configure(
                text="Both checkpoint hashes recorded"
            )
        else:
            evidence = self.evidence[selection]
            self.stat_labels["model"].configure(text=selection)
            self.stat_labels["head"].configure(
                text=self.head_labels[selection]
            )
            self.stat_labels["hash"].configure(
                text=evidence["sha256"][:20] + "..."
            )

        self.model_selector.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.field_start_button.configure(state="disabled")
        self.field_stop_button.configure(state="normal")
        if len(names) == 2:
            self.capture_button.configure(state="normal")

        self.status_var.set(
            f"Camera running: {selection} | {resolution_text}")

        self.worker = threading.Thread(
            target=self._camera_loop,
            daemon=True,
        )
        self.worker.start()

    def stop_camera(self):
        if not self.running:
            return

        self.running = False
        self.status_var.set("Stopping camera...")
        # The camera worker releases the camera in its finally block.

    def _camera_loop(self):
        grabber = None

        try:
            if self.camera is None:
                return

            grabber = LatestFrameGrabber(self.camera).start()
            last_id = 0

            while self.running:
                frame, last_id = grabber.read_new(last_id)
                ok = frame is not None

                if not ok:
                    self.root.after(
                        0,
                        lambda: self.status_var.set(
                            "Unable to read the webcam frame."
                        ),
                    )
                    break

                if len(self.models) == 2:
                    try:
                        request = self.capture_queue.get_nowait()
                    except queue.Empty:
                        request = None

                    if request is not None:
                        self._process_field_capture(frame, request)

                    self._process_comparison_frame(frame)
                else:
                    self._process_single_frame(frame)

                self.frame_index += 1

        except Exception as exc:
            error_text = str(exc)
            self.root.after(
                0,
                lambda error=error_text: self._show_worker_error(error),
            )

        finally:
            self.running = False

            if grabber is not None:
                grabber.stop()

            with self.lock:
                if self.camera is not None:
                    self.camera.release()
                    self.camera = None

            append_jsonl({
                "record_type": "session_end",
                "session_id": self.session_id,
                "frames": self.frame_index,
                "ended_at": timestamp_now(),
            })

            self.root.after(0, self._camera_stopped)

    def _show_worker_error(self, error):
        if not self.closing:
            messagebox.showerror("Camera processing error", error)

    # --------------------------------------------------------
    # SINGLE-MODEL DETECTION
    # --------------------------------------------------------

    def _process_single_frame(self, frame):
        name = self.model_name
        conf = self.current_conf

        result = self._predict(name, frame, conf)
        pre, model_ms, post, total_ms = speed_breakdown(result)

        sticker_ids, blur_ids = self.class_ids[name]
        stickers, privacy = split_detections(result, sticker_ids, blur_ids)
        count = len(stickers)

        now = time.time()

        if stickers and now - self.last_saved_time >= LOG_INTERVAL_S:
            self._save_history(name, stickers, conf, pre, model_ms, post,
                               total_ms)
            self.last_saved_time = now
            self.root.after(0, self._history_count_changed)

        annotated = draw_frame(frame, stickers, privacy)
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(annotated)
        image.thumbnail((850, 560))

        fps = 1000.0 / total_ms if total_ms > 0 else 0.0

        self.root.after(
            0,
            lambda img=image, c=count, p=pre, m=model_ms, q=post,
                   f=fps, model_name=name:
                self._update_single_view(img, c, p, m, q, f, model_name),
        )

    def _save_history(self, name, stickers, conf, pre, model_ms, post,
                      total_ms):
        """One history row per detected sticker."""
        evidence = self.evidence.get(name, {})

        for *_box, confidence, class_name in stickers:
            append_history({
                "timestamp": timestamp_now(),
                "session_id": self.session_id,
                "model": name,
                "inference_head": self.head_labels.get(name, ""),
                "class_name": class_name,
                "confidence": round(confidence, 5),
                "conf_threshold": round(conf, 3),
                "preprocess_ms": round(pre, 3),
                "model_ms": round(model_ms, 3),
                "postprocess_ms": round(post, 3),
                "inference_ms": round(total_ms, 3),
                "frame_count": len(stickers),
                "checkpoint_path": evidence.get("checkpoint_path", ""),
                "checkpoint_sha256": evidence.get("sha256", ""),
                "device": evidence.get("device", ""),
            })

    def _history_count_changed(self):
        self.history_saves = getattr(self, "history_saves", 0) + 1
        if self.tabs.select() == str(self.history_tab):
            self.refresh_history()

    def _update_single_view(self, image, count, pre, model_ms, post, fps,
                            name):
        if self.closing:
            return

        self.photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self.photo, text="")

        self.stat_labels["model"].configure(text=name)
        self.stat_labels["head"].configure(
            text=self.head_labels.get(name, "—")
        )
        self.stat_labels["count"].configure(text=str(count))
        self.stat_labels["inference"].configure(
            text=f"{pre:.1f} / {model_ms:.1f} / {post:.1f}"
        )
        self.stat_labels["fps"].configure(text=f"{fps:.2f}")

        evidence = self.evidence.get(name, {})
        self.stat_labels["hash"].configure(
            text=evidence.get("sha256", "—")[:20] + "..."
        )

        self.status_var.set(
            f"{name}: {count} sticker(s), "
            f"{pre + model_ms + post:.2f} ms per image."
        )

    # --------------------------------------------------------
    # TWO-MODEL COMPARISON
    # --------------------------------------------------------

    def _process_comparison_frame(self, frame):
        annotated = {}
        frame_stats = {}
        frame_stickers = {}
        conf = self.current_conf

        # Alternate which model runs first, so neither is always favoured.
        order = (
            ("YOLO26n", "YOLO11n")
            if self.frame_index % 2 == 0
            else ("YOLO11n", "YOLO26n")
        )

        for position, name in enumerate(order, start=1):
            result = self._predict(name, frame.copy(), conf)
            pre, model_ms, post, total_ms = speed_breakdown(result)

            sticker_ids, blur_ids = self.class_ids[name]
            stickers, privacy = split_detections(
                result, sticker_ids, blur_ids
            )
            count = len(stickers)
            confidences = [s[4] for s in stickers]

            sample = {
                "record_type": "comparison_frame",
                "timestamp": timestamp_now(),
                "session_id": self.session_id,
                "frame_index": self.frame_index,
                "model": name,
                "inference_head": self.head_labels[name],
                "run_position": position,
                "conf_threshold": round(conf, 3),
                "count": count,
                "detected": int(count > 0),
                "avg_confidence": (
                    sum(confidences) / len(confidences)
                    if confidences else 0.0
                ),
                "preprocess_ms": pre,
                "model_ms": model_ms,
                "postprocess_ms": post,
                "inference_ms": total_ms,
            }

            self.comparison_samples[name].append(sample)
            append_jsonl(sample)       # saved immediately
            frame_stats[name] = sample
            frame_stickers[name] = (stickers, pre, model_ms, post,
                                    total_ms)

            image = draw_frame(frame, stickers, privacy)
            height, width = image.shape[:2]
            image = cv2.resize(image, (480, max(1, round(480 * height / width))))

            cv2.putText(
                image,
                f"{name} | {total_ms:.1f} ms | {count} sticker(s)",
                (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            annotated[name] = image

        # History: same rule as single-model mode (at most one save per
        # second), with both models' detections from this same frame.
        now = time.time()
        if (any(item[0] for item in frame_stickers.values())
                and now - self.last_saved_time >= LOG_INTERVAL_S):
            for name, (stickers, pre, model_ms, post, total_ms) in \
                    frame_stickers.items():
                self._save_history(name, stickers, conf, pre, model_ms,
                                   post, total_ms)
            self.last_saved_time = now
            self.root.after(0, self._history_count_changed)

        # Display order is always YOLO26n left, YOLO11n right.
        combined = cv2.hconcat([annotated["YOLO26n"], annotated["YOLO11n"]])
        combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(combined)
        image.thumbnail((1000, 560))

        self.root.after(
            0,
            lambda img=image, stats=frame_stats:
                self._update_comparison_view(img, stats),
        )

    def _update_comparison_view(self, image, stats):
        if self.closing:
            return

        self.photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self.photo, text="")

        if self.tabs.select() == str(self.field_tab):
            preview = image.copy()
            preview.thumbnail((800, 300))
            self.field_photo = ImageTk.PhotoImage(preview)
            self.field_live_label.configure(image=self.field_photo, text="")

        y26 = stats["YOLO26n"]
        y11 = stats["YOLO11n"]

        self.stat_labels["model"].configure(text="Compare both")
        self.stat_labels["count"].configure(
            text=f"26n: {y26['count']} | 11n: {y11['count']}"
        )
        self.stat_labels["inference"].configure(
            text=(
                f"26n: {y26['preprocess_ms']:.1f} / "
                f"{y26['model_ms']:.1f} / {y26['postprocess_ms']:.1f}\n"
                f"11n: {y11['preprocess_ms']:.1f} / "
                f"{y11['model_ms']:.1f} / {y11['postprocess_ms']:.1f}"
            )
        )

        fps_26 = (
            1000.0 / y26["inference_ms"] if y26["inference_ms"] else 0.0
        )
        fps_11 = (
            1000.0 / y11["inference_ms"] if y11["inference_ms"] else 0.0
        )
        self.stat_labels["fps"].configure(
            text=f"26n: {fps_26:.1f} | 11n: {fps_11:.1f}"
        )

        self.status_var.set(
            "Both models are processing the same camera frame. "
            "Stop the camera to see the summary."
        )

    def show_comparison_summary(self):
        for item in self.comparison_table.get_children():
            self.comparison_table.delete(item)

        self.chart.delete("all")

        summaries = {}

        for name, samples in self.comparison_samples.items():
            if not samples:
                continue

            total = len(samples)
            detected_frames = sum(s["detected"] for s in samples)
            avg_objects = sum(s["count"] for s in samples) / total

            confidence_samples = [
                s["avg_confidence"] for s in samples if s["detected"]
            ]
            avg_confidence = (
                sum(confidence_samples) / len(confidence_samples)
                if confidence_samples else 0.0
            )

            avg_model_ms = sum(s["model_ms"] for s in samples) / total
            avg_ms = sum(s["inference_ms"] for s in samples) / total
            fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

            summaries[name] = {
                "frames": total,
                "detected_frames": detected_frames,
                "avg_objects": avg_objects,
                "avg_confidence": avg_confidence,
                "avg_model_ms": avg_model_ms,
                "avg_ms": avg_ms,
                "fps": fps,
            }

            self.comparison_table.insert(
                "",
                "end",
                values=(
                    name,
                    self.head_labels.get(name, "—"),
                    total,
                    detected_frames,
                    f"{avg_objects:.2f}",
                    f"{avg_confidence * 100:.1f}%",
                    f"{avg_model_ms:.2f}",
                    f"{avg_ms:.2f}",
                    f"{fps:.2f}",
                ),
            )

        if not summaries:
            self.comparison_summary.configure(
                text="No comparison frames were recorded."
            )
            return

        self.comparison_heading.configure(
            text=f"Last comparison session: {self.session_id}"
        )

        if len(summaries) == 2:
            y26 = summaries["YOLO26n"]
            y11 = summaries["YOLO11n"]

            if y26["avg_ms"] < y11["avg_ms"]:
                faster = "YOLO26n"
            elif y11["avg_ms"] < y26["avg_ms"]:
                faster = "YOLO11n"
            else:
                faster = "Both models have equal measured time"

            self.comparison_summary.configure(
                text=(
                    f"Faster end-to-end time in this session: {faster}.\n"
                    f"YOLO26n found a sticker in "
                    f"{y26['detected_frames']}/{y26['frames']} frames.\n"
                    f"YOLO11n found a sticker in "
                    f"{y11['detected_frames']}/{y11['frames']} frames.\n\n"
                    "These are webcam observations, not accuracy scores. "
                    "Average confidence is not accuracy. Formal precision, "
                    "recall, and mAP require a labeled test set "
                    "(see the evaluation scripts)."
                )
            )

            width = max(600, self.chart.winfo_width())
            max_ms = max(y26["avg_ms"], y11["avg_ms"]) or 1.0

            for index, name in enumerate(("YOLO26n", "YOLO11n")):
                summary = summaries[name]
                y = 20 + index * 43

                self.chart.create_text(
                    10, y + 10,
                    anchor="w",
                    text=name,
                    fill="black",
                )

                bar_width = (width - 240) * summary["avg_ms"] / max_ms

                self.chart.create_rectangle(
                    100, y, 100 + bar_width, y + 22,
                    fill="#3678bf" if index == 0 else "#e18b35",
                    outline="",
                )

                self.chart.create_text(
                    108 + bar_width, y + 11,
                    anchor="w",
                    text=f"{summary['avg_ms']:.2f} ms",
                    fill="black",
                )

        else:
            self.comparison_summary.configure(
                text=(
                    "Only one model has comparison measurements. "
                    "Select Compare both and run another session."
                )
            )

        self.tabs.select(self.compare_tab)

    def export_comparison(self):
        rows = [
            sample
            for samples in self.comparison_samples.values()
            for sample in samples
        ]

        if not rows:
            messagebox.showinfo(
                "No comparison data",
                "Run Compare both mode and stop the camera first.",
            )
            return

        destination = filedialog.asksaveasfilename(
            title="Export comparison measurements",
            defaultextension=".csv",
            initialfile="webcam_model_comparison.csv",
            filetypes=[("CSV files", "*.csv")],
        )

        if not destination:
            return

        try:
            pd.DataFrame(rows).to_csv(destination, index=False)

            messagebox.showinfo(
                "Export complete",
                f"Comparison data saved to:\n{destination}",
            )

        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    def refresh_history(self):
        if not hasattr(self, "history_table"):
            return

        for item in self.history_table.get_children():
            self.history_table.delete(item)

        try:
            records, skipped = read_history()

            self.history_count_label.configure(
                text=f"Saved detections: {len(records)}"
            )

            for record in reversed(records[-2000:]):
                self.history_table.insert(
                    "",
                    "end",
                    values=tuple(
                        record.get(column, "")
                        for column in HISTORY_COLUMNS
                    ),
                )

            if skipped:
                self.status_var.set(
                    f"History loaded. Skipped {skipped} malformed row(s). "
                    "Original history file preserved."
                )

        except Exception as exc:
            messagebox.showerror("History error", str(exc))

    def export_history(self):
        try:
            records, skipped = read_history()
        except Exception as exc:
            messagebox.showerror("History error", str(exc))
            return

        if not records:
            messagebox.showinfo(
                "No history",
                "There are no valid detection records to export.",
            )
            return

        destination = filedialog.asksaveasfilename(
            title="Export detection history",
            defaultextension=".csv",
            initialfile="live_detection_history_export.csv",
            filetypes=[("CSV files", "*.csv")],
        )

        if not destination:
            return

        try:
            with open(
                destination, "w", newline="", encoding="utf-8-sig"
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=HISTORY_COLUMNS,
                    extrasaction="ignore",
                )
                writer.writeheader()

                for record in records:
                    writer.writerow({
                        column: record.get(column, "")
                        for column in HISTORY_COLUMNS
                    })

            messagebox.showinfo(
                "Export complete",
                f"History exported to:\n{destination}\n\n"
                f"Skipped malformed rows: {skipped}",
            )

        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    def export_evidence(self):
        if not self.evidence:
            messagebox.showinfo(
                "No model evidence",
                "Start the camera to load a model before exporting evidence.",
            )
            return

        destination = filedialog.asksaveasfilename(
            title="Export model evidence",
            defaultextension=".json",
            initialfile="model_checkpoint_evidence.json",
            filetypes=[("JSON files", "*.json")],
        )

        if not destination:
            return

        try:
            Path(destination).write_text(
                json.dumps(self.evidence, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            messagebox.showinfo(
                "Export complete",
                f"Model evidence saved to:\n{destination}",
            )

        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    # --------------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------------

    def _camera_stopped(self):
        if self.closing:
            return

        self.running = False

        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.model_selector.configure(state="readonly")
        self.field_start_button.configure(state="normal")
        self.field_stop_button.configure(state="disabled")
        self.capture_button.configure(state="disabled")
        self.field_auto_var.set(False)
        self.field_live_label.configure(image="", text="Camera stopped")
        self.field_photo = None
        self.refresh_field_summary()

        self.status_var.set(
            "Camera stopped. Live results are separate from formal benchmarks."
        )

        if len(self.models) == 2:
            self.show_comparison_summary()

        self.refresh_history()

    def close(self):
        self.closing = True
        self.running = False
        self._wait_for_worker()

    def _wait_for_worker(self):
        if self.worker is not None and self.worker.is_alive():
            self.root.after(100, self._wait_for_worker)
            return

        with self.lock:
            if self.camera is not None:
                self.camera.release()
                self.camera = None

        self.root.destroy()


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = StickerDetectionApp(root)
    root.mainloop()