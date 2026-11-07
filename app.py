"""
PSU Vehicle Sticker Detection - camera app (live demo and field test).

Two tabs:
  Live demo   YOLO26n and YOLO11n run side by side on the webcam, to show
              the panel the models working. Nothing is saved.
  Field test  The same live view, plus a Capture button. Each capture saves
              the camera photo, runs both models on that exact photo, and
              writes one row to research_outputs/field_test/field_log.csv
              (the distance and both models' results). These
              photos are labelled later and checked by evaluate_field.py
              (Test 2 in the dashboard).

The benchmark results in the thesis come from the evaluation scripts, not
from this live view.

Usage (from the project folder, with the venv active):
    python app.py
"""

import csv
import json
import os
import queue
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
FINAL_RUNS_DIR = BASE_DIR / "runs" / "detect" / "carl-andre-de-castro" / "runs"

MODEL_PATHS = {
    "YOLO26n": FINAL_RUNS_DIR / "yolo26n_final_70_15_15" / "weights" / "best.pt",
    "YOLO11n": FINAL_RUNS_DIR / "yolo11n_final_70_15_15-2" / "weights" / "best.pt",
}
MODELS = tuple(MODEL_PATHS)

# YOLO26n runs NMS-free (nms=False); YOLO11n uses its normal NMS.
PREDICT_ARGS = {"YOLO26n": {"nms": False}, "YOLO11n": {}}

CONF = 0.25              # same "detected" threshold as every test
IMAGE_SIZE = 640         # same input size as training and testing
WARMUP_FRAMES = 10       # blank frames run before the camera starts

CAMERA_INDEX = 0
# 640 x 480 keeps the view smooth; the models resize every frame to 640
# pixels anyway, so a bigger camera picture would only slow the app down.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Field test files
FIELD_DIR = BASE_DIR / "research_outputs" / "field_test"
FIELD_IMAGE_DIR = FIELD_DIR / "images"          # clean photos, to label
FIELD_ANNOTATED_DIR = FIELD_DIR / "annotated"   # both models' boxes
FIELD_DISCARD_DIR = FIELD_DIR / "_discarded"    # captures removed by Undo
FIELD_LOG = FIELD_DIR / "field_log.csv"

# Field test distances. 5 m is the baseline (the training photos were
# all taken at 5 m or closer).
DISTANCES = ["5", "6", "7", "8", "9", "10"]

# The same for every capture: daylight, a parked vehicle that has a
# sticker, nothing covering it. (Lighting, motion blur and occlusion are
# tested in Test 3 instead; at dusk and night, vehicle lights blocked the
# stickers in our trial captures.) Still written to the log, so
# evaluate_field.py reads the same columns.
FIXED = {"lighting": "daylight", "motion": "parked", "occlusion": "none",
         "sticker_present": "yes"}

FIELD_COLUMNS = (
    ["capture_id", "timestamp", "image_file", "annotated_file",
     "distance_m", "lighting", "motion", "occlusion", "sticker_present",
     "notes", "frame_width", "frame_height", "conf_threshold"]
    + [f"{model}_{field}" for model in MODELS
       for field in ("detected", "count", "max_conf", "boxes_xyxy",
                     "inference_ms")]
)

# Colours (same as the dashboard): YOLO26n blue, YOLO11n orange.
BOX_COLOURS = {"YOLO26n": (214, 120, 42), "YOLO11n": (52, 104, 235)}  # BGR
DARK = "#0b0b0b"
SURFACE = "#fcfcfb"
MUTED = "#52514e"


# ============================================================
# FIELD LOG
# ============================================================

def read_field_log():
    """All capture rows (list of dicts); empty list if there is no log."""
    if not FIELD_LOG.is_file() or FIELD_LOG.stat().st_size == 0:
        return []
    with FIELD_LOG.open(newline="", encoding="utf-8-sig") as file:
        return [row for row in csv.DictReader(file) if row.get("capture_id")]


def write_field_rows(rows):
    """Rewrite the whole log (used by Undo and when the columns change)."""
    FIELD_DIR.mkdir(parents=True, exist_ok=True)
    temporary = FIELD_LOG.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELD_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, FIELD_LOG)


def append_field_row(row):
    """Add one capture. A log written by an older app version is converted
    to the current columns first, so no capture is lost."""
    FIELD_DIR.mkdir(parents=True, exist_ok=True)
    if FIELD_LOG.is_file() and FIELD_LOG.stat().st_size > 0:
        with FIELD_LOG.open(newline="", encoding="utf-8-sig") as file:
            header = next(csv.reader(file), [])
        if header != FIELD_COLUMNS:
            write_field_rows(read_field_log())
    new_file = not FIELD_LOG.is_file() or FIELD_LOG.stat().st_size == 0
    with FIELD_LOG.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELD_COLUMNS,
                                extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def summarize_by_distance(rows):
    """
    Captures per distance and in how many of them each model found the
    sticker. Only a quick check while capturing; the real Test 2 results
    come from evaluate_field.py after the photos are labelled.
    """
    def order(distance):
        try:
            return float(distance)
        except ValueError:
            return 999.0

    summary = []
    for distance in sorted({r.get("distance_m", "") for r in rows},
                           key=order):
        group = [r for r in rows if r.get("distance_m", "") == distance]
        item = {"distance": distance, "captures": len(group)}
        for model in MODELS:
            item[f"{model}_found"] = sum(
                r.get(f"{model}_detected") == "1" for r in group)
        summary.append(item)
    return summary


# ============================================================
# DETECTION HELPERS
# ============================================================

def sticker_class_ids(names):
    """Class ids whose name contains 'sticker' (or the only class)."""
    ids = {int(i) for i, n in names.items() if "sticker" in str(n).lower()}
    if not ids and len(names) == 1:
        ids = {int(next(iter(names)))}
    return ids


def find_stickers(result, sticker_ids):
    """List of (x1, y1, x2, y2, confidence) for the sticker boxes."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    found = []
    for (x1, y1, x2, y2), conf, cls in zip(
            boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy().astype(int)):
        if cls in sticker_ids:
            found.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
    return found


def total_ms(result):
    """Ultralytics' own time per image: prepare + model + cleanup (ms)."""
    speed = getattr(result, "speed", {}) or {}
    return sum(float(speed.get(k) or 0.0)
               for k in ("preprocess", "inference", "postprocess"))


def draw_panel(frame, name, stickers, ms=None):
    """One model's view: its sticker boxes and a title line."""
    image = frame.copy()
    colour = BOX_COLOURS[name]
    for x1, y1, x2, y2, conf in stickers:
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 3)
        cv2.putText(image, f"sticker {conf:.2f}", (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    title = f"{name}: {len(stickers)} sticker(s)"
    if ms is not None:
        title += f"  |  {ms:.1f} ms"
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (24, 24, 24), -1)
    cv2.putText(image, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)
    return image


def encode_image(path, image, quality=95):
    """Save a JPG (cv2.imwrite fails on some Windows folder names)."""
    ok, buffer = cv2.imencode(".jpg", image,
                              [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Could not encode image {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(buffer.tobytes())


class LatestFrameGrabber:
    """
    Reads the webcam in its own thread and keeps only the newest frame,
    so the picture on screen never falls behind real life while the
    models are busy. Old frames are simply dropped.
    """

    def __init__(self, camera):
        self.camera = camera
        self.frame = None
        self.frame_id = 0
        self.failed = False
        self.stopped = False
        self.condition = threading.Condition()
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
                or self.stopped, timeout=timeout)
            if self.frame_id == last_id:
                return None, last_id
            return self.frame, self.frame_id

    def stop(self):
        self.stopped = True
        with self.condition:
            self.condition.notify_all()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)


# ============================================================
# THE WINDOW
# ============================================================

class StickerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PSU Vehicle Sticker Detection - YOLO26n vs YOLO11n")
        self.root.geometry("1180x780")
        self.root.minsize(980, 660)
        self.root.configure(bg=SURFACE)

        self.models, self.sticker_ids = {}, {}
        self.camera = None
        self.worker = None
        self.running = False
        self.closing = False
        self.frame_index = 0
        self.capture_count = 0
        self.capture_queue = queue.Queue()
        self.photos = {}                      # keeps Tk images alive

        self.distance_var = tk.StringVar(value=DISTANCES[0])
        self.notes_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Click Start camera.")

        self._style()
        self._build()
        self.refresh_field_table()
        self.root.bind("<space>", self._space_pressed)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # ---------------- layout ----------------

    def _style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=SURFACE, font=("Segoe UI", 10))
        style.configure("TNotebook.Tab", padding=(18, 8),
                        font=("Segoe UI", 11))
        style.configure("Big.TButton", font=("Segoe UI", 11, "bold"),
                        padding=(16, 8))
        style.configure("TButton", padding=(12, 6))
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Card.TLabelframe", background="#ffffff")
        style.configure("Treeview", rowheight=24)
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

    def _build(self):
        header = tk.Frame(self.root, bg=DARK, height=50)
        header.pack(fill="x")
        tk.Label(header, text="PSU Vehicle Sticker Detection  ·  YOLO26n "
                 "vs YOLO11n", bg=DARK, fg="#ffffff",
                 font=("Segoe UI", 14, "bold")).pack(side="left", padx=16,
                                                     pady=10)
        tk.Label(header, text="Research demonstrator, not a deployed system",
                 bg=DARK, fg="#bdbcb8", font=("Segoe UI", 9)).pack(
            side="right", padx=16)

        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=12, pady=(10, 0))
        self.live_tab = ttk.Frame(self.tabs, padding=12)
        self.field_tab = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(self.live_tab, text="  Live demo  ")
        self.tabs.add(self.field_tab, text="  Field test  ")
        self._build_live_tab()
        self._build_field_tab()

        ttk.Label(self.root, textvariable=self.status_var,
                  style="Muted.TLabel", padding=(14, 6)).pack(fill="x")

    def _camera_buttons(self, parent):
        """Start / Stop buttons; the same camera serves both tabs."""
        start = ttk.Button(parent, text="Start camera", style="Big.TButton",
                           command=self.start_camera)
        start.pack(side="left")
        stop = ttk.Button(parent, text="Stop camera", command=self.stop_camera,
                          state="disabled")
        stop.pack(side="left", padx=8)
        return start, stop

    # ---------------- live demo tab ----------------

    def _build_live_tab(self):
        top = ttk.Frame(self.live_tab)
        top.pack(fill="x")
        self.live_start, self.live_stop = self._camera_buttons(top)
        ttk.Label(top, text="YOLO26n on the left (blue), YOLO11n on the "
                  "right (orange). Live view only: nothing is saved.",
                  style="Muted.TLabel").pack(side="left", padx=14)

        self.live_view = ttk.Label(self.live_tab, anchor="center",
                                   text="Camera stopped",
                                   background="#e9e8e4")
        self.live_view.pack(fill="both", expand=True, pady=10)

        cards = ttk.Frame(self.live_tab)
        cards.pack(fill="x")
        self.live_stats = {}
        for name in MODELS:
            box = ttk.LabelFrame(cards, text=f"  {name}  ", padding=10)
            box.pack(side="left", fill="x", expand=True,
                     padx=(0, 10) if name == MODELS[0] else 0)
            label = ttk.Label(box, text="Stickers found:  —\nHighest "
                              "confidence:  —\nTime per image:  —",
                              font=("Segoe UI", 11), justify="left")
            label.pack(anchor="w")
            self.live_stats[name] = label

    # ---------------- field test tab ----------------

    def _build_field_tab(self):
        ttk.Label(self.field_tab, text="1. Start the camera.   2. Choose the "
                  "distance.   3. Press Capture (or the Space bar). Every "
                  "capture: daylight, parked vehicle with a sticker, nothing "
                  "covering it, confidence 0.25.", style="Muted.TLabel",
                  wraplength=1100).pack(anchor="w")

        conditions = ttk.Frame(self.field_tab)
        conditions.pack(fill="x", pady=(8, 4))
        box = ttk.Frame(conditions)
        box.pack(side="left", padx=(0, 16))
        ttk.Label(box, text="Distance (m)").pack(anchor="w")
        ttk.Combobox(box, textvariable=self.distance_var, values=DISTANCES,
                     width=8, state="readonly").pack(anchor="w")
        box = ttk.Frame(conditions)
        box.pack(side="left", fill="x", expand=True)
        ttk.Label(box, text="Notes (optional)").pack(anchor="w")
        ttk.Entry(box, textvariable=self.notes_var).pack(fill="x")

        actions = ttk.Frame(self.field_tab)
        actions.pack(fill="x", pady=8)
        self.field_start, self.field_stop = self._camera_buttons(actions)
        self.capture_button = ttk.Button(
            actions, text="Capture  (Space)", style="Big.TButton",
            command=self.request_capture, state="disabled")
        self.capture_button.pack(side="left", padx=(16, 8))
        ttk.Button(actions, text="Undo last capture",
                   command=self.undo_last_capture).pack(side="left")
        ttk.Button(actions, text="Open field folder",
                   command=self.open_field_folder).pack(side="right")

        middle = ttk.Frame(self.field_tab)
        middle.pack(fill="both", expand=True)
        self.field_view = ttk.Label(middle, anchor="center",
                                    text="Camera stopped",
                                    background="#e9e8e4")
        self.field_view.pack(side="left", fill="both", expand=True,
                             padx=(0, 10))
        info = ttk.LabelFrame(middle, text="  Last capture  ", padding=10,
                              width=280)
        info.pack(side="right", fill="y")
        info.pack_propagate(False)
        self.count_label = ttk.Label(info, text="Captures saved: 0",
                                     font=("Segoe UI", 11, "bold"))
        self.count_label.pack(anchor="w")
        self.last_label = ttk.Label(info, text="No capture yet.",
                                    wraplength=255, justify="left")
        self.last_label.pack(anchor="w", pady=(8, 0))

        table_box = ttk.LabelFrame(
            self.field_tab, text="  Quick check per distance (captures in "
            "which each model found the sticker)  ", padding=6)
        table_box.pack(fill="x", pady=(10, 0))
        columns = ("distance", "captures", "f26", "f11")
        headings = ("Distance", "Captures", "YOLO26n found", "YOLO11n found")
        self.field_table = ttk.Treeview(table_box, columns=columns,
                                        show="headings", height=6)
        for key, text in zip(columns, headings):
            self.field_table.heading(key, text=text)
            self.field_table.column(key, width=200, anchor="center",
                                    stretch=True)
        self.field_table.pack(fill="x")

    # ---------------- camera ----------------

    def _load_models(self):
        blank = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        for name, path in MODEL_PATHS.items():
            if name in self.models:
                continue
            if not path.is_file():
                raise FileNotFoundError(f"{name} model not found:\n{path}")
            self.status_var.set(f"Loading {name}...")
            self.root.update_idletasks()
            model = YOLO(str(path))
            ids = sticker_class_ids(model.names)
            if not ids:
                raise ValueError(f"{name}: no sticker class in {model.names}")
            self.models[name], self.sticker_ids[name] = model, ids
            for _ in range(WARMUP_FRAMES):          # warm-up, not timed
                self._predict(name, blank)

    def _predict(self, name, frame):
        return self.models[name].predict(
            source=frame, conf=CONF, imgsz=IMAGE_SIZE, device=DEVICE,
            verbose=False, **PREDICT_ARGS[name])[0]

    def _set_buttons(self, running):
        for button in (self.live_start, self.field_start):
            button.configure(state="disabled" if running else "normal")
        for button in (self.live_stop, self.field_stop, self.capture_button):
            button.configure(state="normal" if running else "disabled")

    def start_camera(self):
        if self.running:
            return
        try:
            self._load_models()
        except Exception as exc:
            messagebox.showerror("Could not load the models", str(exc))
            self.status_var.set("Model loading failed.")
            return

        camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera.release()
            messagebox.showerror("Camera error", "Could not open the webcam. "
                                 "Close other apps using the camera and try "
                                 "again.")
            return
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        while not self.capture_queue.empty():        # drop old requests
            self.capture_queue.get_nowait()
        self.camera = camera
        self.running = True
        self._set_buttons(True)
        self.status_var.set("Camera running.")
        self.worker = threading.Thread(target=self._camera_loop, daemon=True)
        self.worker.start()

    def stop_camera(self):
        self.running = False
        self.status_var.set("Stopping camera...")

    def _camera_loop(self):
        grabber = None
        try:
            grabber = LatestFrameGrabber(self.camera).start()
            last_id = 0
            while self.running:
                frame, last_id = grabber.read_new(last_id)
                if frame is None:
                    self.root.after(0, lambda: self.status_var.set(
                        "The webcam stopped sending pictures."))
                    break
                try:
                    request = self.capture_queue.get_nowait()
                except queue.Empty:
                    request = None
                if request is not None:
                    self._save_capture(frame, request)
                self._show_frame(frame)
                self.frame_index += 1
        except Exception as exc:
            message = str(exc)
            self.root.after(0, lambda: messagebox.showerror(
                "Camera error", message))
        finally:
            self.running = False
            if grabber is not None:
                grabber.stop()
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            self.root.after(0, self._camera_stopped)

    def _run_both(self, frame):
        """Both models on the same frame. The order swaps every frame so
        neither model always runs first."""
        order = MODELS if self.frame_index % 2 == 0 else MODELS[::-1]
        results = {}
        for name in order:
            result = self._predict(name, frame)
            results[name] = (find_stickers(result, self.sticker_ids[name]),
                             total_ms(result))
        return results

    def _show_frame(self, frame):
        results = self._run_both(frame)
        panels = [draw_panel(frame, name, *results[name]) for name in MODELS]
        picture = Image.fromarray(cv2.cvtColor(cv2.hconcat(panels),
                                               cv2.COLOR_BGR2RGB))
        self.root.after(0, lambda: self._update_view(picture, results))

    def _update_view(self, picture, results):
        if self.closing:
            return
        on_field = self.tabs.select() == str(self.field_tab)
        target = self.field_view if on_field else self.live_view
        width = max(400, target.winfo_width() - 8)
        height = max(250, target.winfo_height() - 8)
        picture.thumbnail((width, height))
        self.photos["view"] = ImageTk.PhotoImage(picture)
        target.configure(image=self.photos["view"], text="")
        for name, (stickers, ms) in results.items():
            best = max((s[4] for s in stickers), default=None)
            self.live_stats[name].configure(text=(
                f"Stickers found:  {len(stickers)}\n"
                f"Highest confidence:  "
                f"{'—' if best is None else f'{best:.2f}'}\n"
                f"Time per image:  {ms:.1f} ms  "
                f"({1000 / ms if ms else 0:.0f} photos per second)"))

    def _camera_stopped(self):
        if self.closing:
            return
        self._set_buttons(False)
        for view in (self.live_view, self.field_view):
            view.configure(image="", text="Camera stopped")
        self.photos.pop("view", None)
        self.status_var.set("Camera stopped.")

    # ---------------- field capture ----------------

    def _space_pressed(self, _event=None):
        """Space = Capture, but only on the Field test tab and not while
        typing in the Notes box."""
        if isinstance(self.root.focus_get(), (ttk.Entry, tk.Entry)):
            return None
        if self.tabs.select() == str(self.field_tab):
            self.request_capture()
            return "break"
        return None

    def request_capture(self):
        if not self.running:
            messagebox.showinfo("Camera not running", "Click Start camera "
                                "first.")
            return
        self.capture_queue.put({
            "distance_m": self.distance_var.get(),
            "notes": self.notes_var.get().strip(),
        })
        self.status_var.set("Capturing...")

    def _save_capture(self, frame, request):
        """Runs in the camera thread on the exact frame that was read:
        saves the clean photo, the photo with both models' boxes, and one
        row in the field log."""
        self.capture_count += 1
        now = datetime.now()
        capture_id = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        stem = f"field_{capture_id}_d{request['distance_m']}m"
        results = self._run_both(frame)

        image_path = FIELD_IMAGE_DIR / f"{stem}.jpg"
        encode_image(image_path, frame)                 # no boxes: to label
        annotated_path = FIELD_ANNOTATED_DIR / f"{stem}_both.jpg"
        encode_image(annotated_path, cv2.hconcat(
            [draw_panel(frame, name, results[name][0]) for name in MODELS]),
            quality=85)

        row = {
            "capture_id": capture_id,
            "timestamp": now.isoformat(timespec="seconds"),
            "image_file": str(image_path.relative_to(FIELD_DIR)),
            "annotated_file": str(annotated_path.relative_to(FIELD_DIR)),
            **request, **FIXED,
            "frame_width": frame.shape[1], "frame_height": frame.shape[0],
            "conf_threshold": CONF,
        }
        for name in MODELS:
            stickers, ms = results[name]
            row[f"{name}_detected"] = int(bool(stickers))
            row[f"{name}_count"] = len(stickers)
            row[f"{name}_max_conf"] = (round(max(s[4] for s in stickers), 5)
                                       if stickers else "")
            row[f"{name}_boxes_xyxy"] = json.dumps(
                [[*s[:4], round(s[4], 5)] for s in stickers])
            row[f"{name}_inference_ms"] = round(ms, 3)
        append_field_row(row)
        self.root.after(0, lambda: self._capture_saved(row))

    def _capture_saved(self, row):
        if self.closing:
            return

        def line(name):
            if row[f"{name}_detected"]:
                return (f"{name}: FOUND (confidence "
                        f"{float(row[f'{name}_max_conf']):.2f})")
            return f"{name}: not found"

        self.last_label.configure(text=(
            f"{row['distance_m']} m\n\n{line('YOLO26n')}\n"
            f"{line('YOLO11n')}\n\n{row['image_file']}"))
        self.status_var.set(f"Saved {row['image_file']}")
        self.refresh_field_table()

    def refresh_field_table(self):
        rows = read_field_log()
        self.count_label.configure(text=f"Captures saved: {len(rows)}")
        self.field_table.delete(*self.field_table.get_children())

        for item in summarize_by_distance(rows):
            total = item["captures"]
            self.field_table.insert("", "end", values=(
                f"{item['distance']} m", total,
                f"{item['YOLO26n_found']} / {total}",
                f"{item['YOLO11n_found']} / {total}"))

    def undo_last_capture(self):
        rows = read_field_log()
        if not rows:
            messagebox.showinfo("Nothing to undo", "No captures yet.")
            return
        last = rows[-1]
        if not messagebox.askyesno(
                "Undo last capture", f"Remove the last capture "
                f"({last['distance_m']} m)?\n\nIts photos "
                "are moved to the _discarded folder, not deleted."):
            return
        FIELD_DISCARD_DIR.mkdir(parents=True, exist_ok=True)
        for key in ("image_file", "annotated_file"):
            source = FIELD_DIR / last.get(key, "")
            if last.get(key) and source.is_file():
                os.replace(source, FIELD_DISCARD_DIR / source.name)
        write_field_rows(rows[:-1])
        self.last_label.configure(text="Last capture removed.")
        self.refresh_field_table()

    def open_field_folder(self):
        FIELD_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(FIELD_DIR)                     # Windows
        except AttributeError:
            messagebox.showinfo("Field folder", str(FIELD_DIR))

    # ---------------- closing ----------------

    def close(self):
        self.closing = True
        self.running = False
        self._wait_for_worker()

    def _wait_for_worker(self):
        if self.worker is not None and self.worker.is_alive():
            self.root.after(100, self._wait_for_worker)
            return
        if self.camera is not None:
            self.camera.release()
        self.root.destroy()


# ============================================================
# START
# ============================================================

def main():
    root = tk.Tk()
    StickerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()