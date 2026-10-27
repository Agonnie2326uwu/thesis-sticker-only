
import csv
import hashlib
import json
import os
import threading
import time
import tkinter as tk

from datetime import datetime
from pathlib import Path
from tkinter import ttk, messagebox, filedialog

import cv2
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

MODEL_PATHS = {
    "YOLO26n": (
        BASE_DIR / "runs" / "detect" / "runs"
        / "yolo26n_sticker_split80_10_10"
        / "weights" / "best.pt"
    ),
    "YOLO11n": (
        BASE_DIR / "runs" / "detect" / "runs"
        / "yolo11n_sticker_split80_10_10"
        / "weights" / "best.pt"
    ),
}

HISTORY_FILE = OUTPUT_DIR / "live_detection_history.csv"
SESSION_FILE = OUTPUT_DIR / "live_detection_sessions.jsonl"

HISTORY_COLUMNS = [
    "timestamp",
    "session_id",
    "model",
    "class_name",
    "confidence",
    "inference_ms",
    "frame_count",
    "checkpoint_path",
    "checkpoint_sha256",
    "device",
]

CAMERA_INDEX = 0
IMAGE_SIZE = 640
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

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

            # Map the existing file's columns by their names.
            header = [str(name).strip() for name in header]

            for row in reader:
                if not row or not any(str(value).strip() for value in row):
                    continue

                # Prevent malformed rows from breaking the history page.
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


def append_history(record):
    """
    Always write exactly the same ten columns.
    If an older history file has a different header, preserve it as a backup.
    """
    if HISTORY_FILE.exists() and HISTORY_FILE.stat().st_size > 0:
        try:
            with HISTORY_FILE.open(
                "r", newline="", encoding="utf-8-sig"
            ) as file:
                existing_header = next(csv.reader(file), [])

            if existing_header != HISTORY_COLUMNS:
                backup = OUTPUT_DIR / (
                    "live_detection_history_backup_"
                    + datetime.now().strftime("%Y%m%d_%H%M%S")
                    + ".csv"
                )
                os.replace(HISTORY_FILE, backup)

        except (OSError, csv.Error, UnicodeError):
            # Do not append into a file whose header cannot be verified.
            backup = OUTPUT_DIR / (
                "live_detection_history_backup_"
                + datetime.now().strftime("%Y%m%d_%H%M%S")
                + ".csv"
            )
            os.replace(HISTORY_FILE, backup)

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


def model_evidence(name, model, checkpoint):
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
        self.model_name = None
        self.session_id = None
        self.photo = None

        self.current_conf = 0.25
        self.lock = threading.Lock()
        self.last_saved_time = 0.0
        self.ui_after_id = None

        self.comparison_samples = {
            name: [] for name in MODEL_PATHS
        }

        self.model_var = tk.StringVar(value="Compare both")
        self.conf_var = tk.DoubleVar(value=0.25)
        self.status_var = tk.StringVar(
            value="Ready. Select a model and start the camera."
        )

        self._build_interface()
        self.refresh_history()

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
            text="Research demonstrator • One webcam",
        ).pack(side="right")

        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=5)

        self.live_tab = ttk.Frame(self.tabs, padding=10)
        self.compare_tab = ttk.Frame(self.tabs, padding=10)
        self.history_tab = ttk.Frame(self.tabs, padding=10)

        self.tabs.add(self.live_tab, text="Live Detection")
        self.tabs.add(self.compare_tab, text="Model Comparison")
        self.tabs.add(self.history_tab, text="History & Model Evidence")

        self._build_live_tab()
        self._build_compare_tab()
        self._build_history_tab()

        ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w",
            padding=5,
        ).pack(fill="x")

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
            ("Checkpoint SHA-256", "hash"),
            ("Detections in current frame", "count"),
            ("Inference time (ms)", "inference"),
            ("Approx. FPS", "fps"),
            ("Detected classes", "classes"),
            ("Device", "device"),
        ]

        for title, key in stat_items:
            ttk.Label(stats_frame, text=title).pack(
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
                "Both models will process the same captured frames. "
                "Stop the camera to view the summary."
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
            "frames",
            "detected_frames",
            "avg_objects",
            "avg_confidence",
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
            "model": ("Model", 105),
            "frames": ("Frames processed", 115),
            "detected_frames": ("Frames with detection", 140),
            "avg_objects": ("Avg objects/frame", 125),
            "avg_confidence": ("Avg confidence", 110),
            "avg_ms": ("Avg inference (ms)", 125),
            "fps": ("Approx. model FPS", 120),
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
            text="Average inference time (lower is faster)",
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
            "class_name": 110,
            "confidence": 90,
            "inference_ms": 95,
            "frame_count": 100,
            "checkpoint_path": 230,
            "checkpoint_sha256": 240,
            "device": 130,
        }

        for key in columns:
            self.history_table.heading(key, text=key.replace("_", " ").title())
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
        self.stat_labels["hash"].configure(text="—")
        self.status_var.set(
            "Selection updated. Start the camera to load the model."
        )

    def _load_model(self, name):
        checkpoint = MODEL_PATHS[name]

        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"{name} checkpoint was not found:\n\n{checkpoint}\n\n"
                "Check that the best.pt file exists at this path."
            )

        model = YOLO(str(checkpoint))
        evidence = model_evidence(name, model, checkpoint)

        self.models[name] = model
        self.evidence[name] = evidence

        append_jsonl({
            "record_type": "model_loaded",
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

            selection = self.model_var.get()

            names = (
                list(MODEL_PATHS.keys())
                if selection == "Compare both"
                else [selection]
            )

            for name in names:
                self.status_var.set(f"Loading {name}...")
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

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        self.camera = camera
        self.running = True
        self.last_saved_time = 0.0

        self.session_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_"
            + selection.replace(" ", "_")
        )

        self.comparison_samples = {
            name: [] for name in names
        }

        if selection == "Compare both":
            self.stat_labels["model"].configure(text="Compare both")
            self.stat_labels["hash"].configure(
                text="Both checkpoint hashes recorded"
            )
        else:
            evidence = self.evidence[selection]
            self.stat_labels["model"].configure(text=selection)
            self.stat_labels["hash"].configure(
                text=evidence["sha256"][:20] + "..."
            )

        self.model_selector.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

        self.status_var.set(f"Camera running: {selection}")

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

        # Do not release the camera from a second thread while read() runs.
        # The camera worker releases it in its finally block.

    def _camera_loop(self):
        try:
            while self.running:
                camera = self.camera

                if camera is None:
                    break

                ok, frame = camera.read()

                if not ok:
                    self.root.after(
                        0,
                        lambda: self.status_var.set(
                            "Unable to read the webcam frame."
                        ),
                    )
                    break

                if len(self.models) == 2:
                    self._process_comparison_frame(frame)
                else:
                    self._process_single_frame(frame)

        except Exception as exc:
            error_text = str(exc)
            self.root.after(
                0,
                lambda error=error_text: self._show_worker_error(error),
            )

        finally:
            self.running = False

            with self.lock:
                if self.camera is not None:
                    self.camera.release()
                    self.camera = None

            self.root.after(0, self._camera_stopped)

    def _show_worker_error(self, error):
        if not self.closing:
            messagebox.showerror("Camera processing error", error)

    # --------------------------------------------------------
    # SINGLE-MODEL DETECTION
    # --------------------------------------------------------

    def _process_single_frame(self, frame):
        name = self.model_name
        model = self.models[name]

        started = time.perf_counter()

        result = model.predict(
            source=frame,
            conf=self.current_conf,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
        )[0]

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        boxes = result.boxes
        count = len(boxes) if boxes is not None else 0

        detections = []

        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                class_name = result.names.get(
                    class_id,
                    str(class_id),
                )

                detections.append({
                    "class_name": str(class_name),
                    "confidence": confidence,
                })

        # Save detected objects at most once per second.
        now = time.time()

        if detections and now - self.last_saved_time >= 1.0:
            evidence = self.evidence[name]

            for detection in detections:
                append_history({
                    "timestamp": timestamp_now(),
                    "session_id": self.session_id,
                    "model": name,
                    "class_name": detection["class_name"],
                    "confidence": round(
                        detection["confidence"], 5
                    ),
                    "inference_ms": round(elapsed_ms, 3),
                    "frame_count": count,
                    "checkpoint_path": evidence["checkpoint_path"],
                    "checkpoint_sha256": evidence["sha256"],
                    "device": evidence["device"],
                })

            self.last_saved_time = now

        annotated = result.plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(annotated)
        image.thumbnail((850, 560))

        fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
        class_text = ", ".join(
            item["class_name"] for item in detections
        ) or "None"

        self.root.after(
            0,
            lambda img=image, c=count, ms=elapsed_ms,
                   f=fps, classes=class_text, model_name=name:
                self._update_single_view(
                    img, c, ms, f, classes, model_name
                ),
        )

    def _update_single_view(
        self, image, count, elapsed_ms, fps, classes, name
    ):
        if self.closing:
            return

        self.photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self.photo, text="")

        self.stat_labels["model"].configure(text=name)
        self.stat_labels["count"].configure(text=str(count))
        self.stat_labels["inference"].configure(
            text=f"{elapsed_ms:.2f}"
        )
        self.stat_labels["fps"].configure(text=f"{fps:.2f}")
        self.stat_labels["classes"].configure(text=classes)

        evidence = self.evidence.get(name, {})
        self.stat_labels["hash"].configure(
            text=evidence.get("sha256", "—")[:20] + "..."
        )

        self.status_var.set(
            f"{name}: {count} object(s), "
            f"{elapsed_ms:.2f} ms measured inference time."
        )

    # --------------------------------------------------------
    # TWO-MODEL COMPARISON
    # --------------------------------------------------------

    def _process_comparison_frame(self, frame):
        annotated_frames = []
        frame_stats = {}

        # The exact same captured frame is passed to both models.
        for name in ("YOLO26n", "YOLO11n"):
            model = self.models[name]

            started = time.perf_counter()

            result = model.predict(
                source=frame.copy(),
                conf=self.current_conf,
                imgsz=IMAGE_SIZE,
                device=DEVICE,
                verbose=False,
            )[0]

            elapsed_ms = (time.perf_counter() - started) * 1000.0

            boxes = result.boxes
            count = len(boxes) if boxes is not None else 0

            confidences = []
            classes = []

            if boxes is not None:
                for box in boxes:
                    confidences.append(float(box.conf[0].item()))

                    class_id = int(box.cls[0].item())
                    classes.append(
                        str(result.names.get(class_id, class_id))
                    )

            avg_confidence = (
                sum(confidences) / len(confidences)
                if confidences else 0.0
            )

            sample = {
                "timestamp": timestamp_now(),
                "session_id": self.session_id,
                "model": name,
                "count": count,
                "avg_confidence": avg_confidence,
                "inference_ms": elapsed_ms,
                "detected": int(count > 0),
            }

            self.comparison_samples[name].append(sample)
            frame_stats[name] = sample

            annotated = result.plot()
            annotated = cv2.resize(annotated, (480, 360))

            cv2.putText(
                annotated,
                f"{name} | {elapsed_ms:.1f} ms | {count} object(s)",
                (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            annotated_frames.append(annotated)

        combined = cv2.hconcat(annotated_frames)
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

        y26 = stats["YOLO26n"]
        y11 = stats["YOLO11n"]

        self.stat_labels["model"].configure(text="Compare both")
        self.stat_labels["count"].configure(
            text=f"26n: {y26['count']} | 11n: {y11['count']}"
        )
        self.stat_labels["inference"].configure(
            text=f"26n: {y26['inference_ms']:.2f} | "
                 f"11n: {y11['inference_ms']:.2f}"
        )

        combined_ms = y26["inference_ms"] + y11["inference_ms"]
        combined_fps = 1000.0 / combined_ms if combined_ms > 0 else 0

        self.stat_labels["fps"].configure(
            text=f"{combined_fps:.2f} combined"
        )
        self.stat_labels["classes"].configure(
            text=(
                f"26n: {y26['count']} object(s) | "
                f"11n: {y11['count']} object(s)"
            )
        )

        self.stat_labels["hash"].configure(
            text="Two checkpoint hashes recorded"
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
                s["avg_confidence"]
                for s in samples
                if s["detected"]
            ]

            avg_confidence = (
                sum(confidence_samples) / len(confidence_samples)
                if confidence_samples else 0.0
            )

            avg_ms = sum(s["inference_ms"] for s in samples) / total
            fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

            summaries[name] = {
                "frames": total,
                "detected_frames": detected_frames,
                "avg_objects": avg_objects,
                "avg_confidence": avg_confidence,
                "avg_ms": avg_ms,
                "fps": fps,
            }

            self.comparison_table.insert(
                "",
                "end",
                values=(
                    name,
                    total,
                    detected_frames,
                    f"{avg_objects:.2f}",
                    f"{avg_confidence * 100:.1f}%",
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
                faster = "Both models have equal measured inference time"

            self.comparison_summary.configure(
                text=(
                    f"Faster measured inference in this session: {faster}.\n"
                    f"YOLO26n detected objects in "
                    f"{y26['detected_frames']}/{y26['frames']} frames.\n"
                    f"YOLO11n detected objects in "
                    f"{y11['detected_frames']}/{y11['frames']} frames.\n\n"
                    "These are webcam observations, not accuracy scores. "
                    "Average confidence is not accuracy. Formal precision, "
                    "recall, and mAP require a labeled test set."
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

                bar_width = (
                    (width - 240) * summary["avg_ms"] / max_ms
                )

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
            # Export only valid rows in the standardized column order.
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

        self.status_var.set(
            "Camera stopped. Live results are separate from formal benchmarks."
        )

        if len(self.models) == 2:
            self.show_comparison_summary()

        self.refresh_history()

    def close(self):
        self.closing = True
        self.running = False

        # Wait for the camera thread to release the webcam.
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
