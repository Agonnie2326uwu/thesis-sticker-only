"""
PSU Vehicle Sticker Detection - Demonstration Interface
Research context: YOLO26n vs YOLO11n vehicle-sticker benchmarking.

This GUI is for live demonstration and provenance logging. It is NOT a
substitute for the paper's controlled test-set/stressor experiments.
"""
import csv
import hashlib
import json
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk, messagebox, filedialog

import cv2
import pandas as pd
import torch
from PIL import Image, ImageTk
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "research_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATHS = {
    "YOLO26n": BASE_DIR / "runs/detect/runs/yolo26n_sticker_split80_10_10/weights/best.pt",
    "YOLO11n": BASE_DIR / "runs/detect/runs/yolo11n_sticker_split80_10_10/weights/best.pt",
}
HISTORY_FILE = OUTPUT_DIR / "live_detection_history.csv"
SESSION_FILE = OUTPUT_DIR / "live_detection_sessions.jsonl"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
CAMERA_INDEX = 0
IMG_SIZE = 640


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_provenance(model_name: str, model: YOLO) -> dict:
    path = MODEL_PATHS[model_name]
    net = getattr(model, "model", None)
    yaml = getattr(net, "yaml", {}) or {}
    try:
        params = sum(p.numel() for p in net.parameters())
    except Exception:
        params = None
    return {
        "display_name": model_name,
        "checkpoint_path": str(path.resolve()),
        "sha256": sha256_file(path),
        "task": getattr(model, "task", "unknown"),
        "yaml_file": yaml.get("yaml_file", ""),
        "scale": yaml.get("scale", ""),
        "parameters": params,
        "class_names": json.dumps(model.names, ensure_ascii=False),
        "ultralytics_version": __import__("ultralytics").__version__,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "verified_at": datetime.now().isoformat(timespec="seconds"),
    }


class StickerDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PSU Vehicle Sticker Detection — Model Demonstrator")
        self.root.geometry("1180x760")
        self.root.minsize(900, 600)

        self.camera = None
        self.running = False
        self.worker = None
        self.model = None
        self.model_name = None
        self.photo = None
        self.lock = threading.Lock()
        self.current_conf = 0.25
        self.last_error = None
        self.session_id = None
        self.session_records = 0
        self.last_saved_detection = 0.0

        self.model_var = tk.StringVar(value="YOLO26n")
        self.conf_var = tk.DoubleVar(value=0.25)
        self.status_var = tk.StringVar(value="Ready — select a model and start the camera.")
        self._build_ui()
        self._refresh_history()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Heading.TLabel", font=("Segoe UI", 17, "bold"))
        style.configure("Metric.TLabel", font=("Segoe UI", 11, "bold"))

        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")
        ttk.Label(top, text="VEHICLE STICKER DETECTION", style="Heading.TLabel").pack(side="left")
        ttk.Label(top, text="Research demonstrator • one camera").pack(side="right")

        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self.live_tab = ttk.Frame(self.tabs, padding=10)
        self.results_tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(self.live_tab, text="Live Detection")
        self.tabs.add(self.results_tab, text="History & Model Evidence")
        self._build_live()
        self._build_results()
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w", padding=5).pack(fill="x")

    def _build_live(self):
        bar = ttk.LabelFrame(self.live_tab, text="Controls", padding=8)
        bar.pack(fill="x", pady=(0, 8))
        ttk.Label(bar, text="Model:").grid(row=0, column=0, padx=4, pady=4)
        self.selector = ttk.Combobox(bar, textvariable=self.model_var, values=list(MODEL_PATHS), state="readonly", width=12)
        self.selector.grid(row=0, column=1, padx=4, pady=4)
        self.selector.bind("<<ComboboxSelected>>", self._model_changed)
        ttk.Label(bar, text="Confidence threshold:").grid(row=0, column=2, padx=(16, 4))
        scale = ttk.Scale(bar, from_=0.10, to=0.90, variable=self.conf_var, orient="horizontal", length=140, command=self._confidence_changed)
        scale.grid(row=0, column=3, padx=4)
        self.conf_label = ttk.Label(bar, text="0.25")
        self.conf_label.grid(row=0, column=4, padx=4)
        self.start_btn = ttk.Button(bar, text="Start Camera", command=self.start)
        self.start_btn.grid(row=0, column=5, padx=(14, 4))
        self.stop_btn = ttk.Button(bar, text="Stop Camera", command=self.stop, state="disabled")
        self.stop_btn.grid(row=0, column=6, padx=4)

        body = ttk.Frame(self.live_tab)
        body.pack(fill="both", expand=True)
        view = ttk.LabelFrame(body, text="Camera feed", padding=6)
        view.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.video = ttk.Label(view, text="Camera stopped\nClick Start Camera", anchor="center", justify="center")
        self.video.pack(fill="both", expand=True)

        metrics = ttk.LabelFrame(body, text="Live statistics", padding=10, width=270)
        metrics.pack(side="right", fill="y")
        metrics.pack_propagate(False)
        self.metrics = {}
        for caption, key in [
            ("Selected model", "model"),
            ("Checkpoint SHA-256", "hash"),
            ("Detections in current frame", "count"),
            ("Inference time (ms)", "ms"),
            ("Approx. FPS", "fps"),
            ("Detected classes", "classes"),
            ("Device", "device"),
        ]:
            ttk.Label(metrics, text=caption).pack(anchor="w", pady=(8, 1))
            label = ttk.Label(metrics, text="—", style="Metric.TLabel", wraplength=240)
            label.pack(anchor="w")
            self.metrics[key] = label
        self.metrics["device"].configure(text=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
        ttk.Button(metrics, text="Open history", command=lambda: self.tabs.select(self.results_tab)).pack(fill="x", pady=(18, 4))
        ttk.Label(metrics, text="For formal results, use the fixed test set and controlled stressor scripts—not webcam counts.", wraplength=240).pack(anchor="w", pady=8)

    def _build_results(self):
        top = ttk.Frame(self.results_tab)
        top.pack(fill="x", pady=(0, 8))
        self.history_label = ttk.Label(top, text="Saved live detections: 0", style="Metric.TLabel")
        self.history_label.pack(side="left")
        ttk.Button(top, text="Refresh", command=self._refresh_history).pack(side="right", padx=4)
        ttk.Button(top, text="Export CSV", command=self.export_csv).pack(side="right", padx=4)
        ttk.Button(top, text="Export model evidence JSON", command=self.export_model_evidence).pack(side="right", padx=4)

        columns = ("timestamp", "session_id", "model", "class_name", "confidence", "inference_ms", "frame_count", "checkpoint_sha256")
        self.table = ttk.Treeview(self.results_tab, columns=columns, show="headings")
        heading = {
            "timestamp": ("Timestamp", 145), "session_id": ("Session", 130), "model": ("Model", 85),
            "class_name": ("Class", 130), "confidence": ("Confidence", 90), "inference_ms": ("Inference ms", 95),
            "frame_count": ("Objects/frame", 100), "checkpoint_sha256": ("Checkpoint SHA-256", 260)
        }
        for col, (label, width) in heading.items():
            self.table.heading(col, text=label)
            self.table.column(col, width=width, anchor="center")
        scroll = ttk.Scrollbar(self.results_tab, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _confidence_changed(self, value):
        self.current_conf = float(value)
        self.conf_label.configure(text=f"{self.current_conf:.2f}")

    def _model_changed(self, _event=None):
        if self.running:
            messagebox.showinfo("Stop camera first", "Stop the camera before switching models.")
            self.model_var.set(self.model_name or "YOLO26n")
            return
        self.model = None
        self.model_name = None
        self.status_var.set("Model selection changed. Start camera to load and verify the checkpoint.")

    def _load_model(self, name):
        path = MODEL_PATHS[name]
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found:\n{path}")
        self.status_var.set(f"Loading and verifying {name}...")
        self.root.update_idletasks()
        model = YOLO(str(path))
        self.model = model
        self.model_name = name
        self.provenance = model_provenance(name, model)
        # Persist provenance so the exact checkpoint can be documented.
        with SESSION_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"record_type": "model_loaded", **self.provenance}, ensure_ascii=False) + "\n")
        self.metrics["model"].configure(text=name)
        self.metrics["hash"].configure(text=self.provenance["sha256"][:20] + "…")
        self.status_var.set(f"{name} loaded. SHA-256 recorded.")

    def start(self):
        if self.running:
            return
        try:
            self._load_model(self.model_var.get())
        except Exception as exc:
            messagebox.showerror("Model verification/loading failed", str(exc))
            return
        cam = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cam.isOpened():
            cam.release()
            messagebox.showerror("Camera error", "Could not open webcam index 0. Close other webcam apps and try again.")
            return
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera = cam
        self.running = True
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + self.model_name
        self.session_records = 0
        self.last_saved_detection = 0
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def _loop(self):
        while self.running:
            ok, frame = self.camera.read()
            if not ok:
                self.root.after(0, lambda: self.status_var.set("Could not read a camera frame."))
                break
            started = time.perf_counter()
            try:
                result = self.model.predict(
                    source=frame, conf=self.current_conf, imgsz=IMG_SIZE,
                    device=DEVICE, verbose=False
                )[0]
            except Exception as exc:
                self.root.after(0, lambda e=str(exc): messagebox.showerror("Detection error", e))
                break
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            boxes = result.boxes
            count = len(boxes) if boxes is not None else 0
            names, confs = [], []
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    names.append(str(result.names.get(cls_id, cls_id)))
                    confs.append(float(box.conf[0].item()))

            if count and (time.time() - self.last_saved_detection) >= 1.0:
                stamp = datetime.now().isoformat(timespec="seconds")
                for name, confidence in zip(names, confs):
                    record = {
                        "timestamp": stamp,
                        "session_id": self.session_id,
                        "model": self.model_name,
                        "class_name": name,
                        "confidence": round(confidence, 5),
                        "inference_ms": round(elapsed_ms, 3),
                        "frame_count": count,
                        "checkpoint_path": self.provenance["checkpoint_path"],
                        "checkpoint_sha256": self.provenance["sha256"],
                        "device": self.provenance["device"],
                    }
                    self._append_record(record)
                self.last_saved_detection = time.time()

            annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
            image = Image.fromarray(annotated)
            image.thumbnail((820, 540))
            photo = ImageTk.PhotoImage(image)
            fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0
            self.root.after(0, lambda p=photo, c=count, ms=elapsed_ms, f=fps, ns=", ".join(names) or "None": self._update(p, c, ms, f, ns))
        self.root.after(0, self._stopped)

    def _append_record(self, record):
        with self.lock:
            new_file = not HISTORY_FILE.exists() or HISTORY_FILE.stat().st_size == 0
            with HISTORY_FILE.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(record.keys()))
                if new_file:
                    writer.writeheader()
                writer.writerow(record)
            self.session_records += 1

    def _update(self, photo, count, ms, fps, classes):
        if not self.running:
            return
        self.photo = photo
        self.video.configure(image=photo, text="")
        self.metrics["count"].configure(text=str(count))
        self.metrics["ms"].configure(text=f"{ms:.2f}")
        self.metrics["fps"].configure(text=f"{fps:.2f}")
        self.metrics["classes"].configure(text=classes)
        self.status_var.set(f"{self.model_name} | {count} object(s) in current frame | {ms:.2f} ms measured for this frame")

    def stop(self):
        self.running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self._stopped()

    def _stopped(self):
        self.running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Camera stopped. Live results are saved separately from formal benchmark results.")
        self._refresh_history()

    def _refresh_history(self):
        if not hasattr(self, "table"):
            return
        for item in self.table.get_children():
            self.table.delete(item)
        if HISTORY_FILE.exists():
            try:
                df = pd.read_csv(HISTORY_FILE).fillna("")
                records = df.tail(2000).to_dict(orient="records")
                self.history_label.configure(text=f"Saved live detections: {len(df)}")
                for r in reversed(records):
                    self.table.insert("", "end", values=(
                        r.get("timestamp", ""), r.get("session_id", ""), r.get("model", ""),
                        r.get("class_name", ""), r.get("confidence", ""), r.get("inference_ms", ""),
                        r.get("frame_count", ""), r.get("checkpoint_sha256", "")
                    ))
            except Exception as exc:
                messagebox.showwarning("History read warning", str(exc))
        else:
            self.history_label.configure(text="Saved live detections: 0")

    def export_csv(self):
        if not HISTORY_FILE.exists() or HISTORY_FILE.stat().st_size == 0:
            messagebox.showinfo("No history", "No live detection records are available yet.")
            return
        destination = filedialog.asksaveasfilename(
            title="Export live detection history", defaultextension=".csv",
            initialfile="live_detection_history.csv", filetypes=[("CSV files", "*.csv")]
        )
        if destination:
            try:
                pd.read_csv(HISTORY_FILE).to_csv(destination, index=False)
                messagebox.showinfo("Export complete", f"Saved to:\n{destination}")
            except Exception as exc:
                messagebox.showerror("Export failed", str(exc))

    def export_model_evidence(self):
        if not getattr(self, "provenance", None):
            messagebox.showinfo("No model loaded", "Start the camera once to load a model and record its provenance.")
            return
        destination = filedialog.asksaveasfilename(
            title="Export checkpoint evidence", defaultextension=".json",
            initialfile=f"{self.model_name}_checkpoint_evidence.json",
            filetypes=[("JSON files", "*.json")]
        )
        if destination:
            Path(destination).write_text(json.dumps(self.provenance, indent=2, ensure_ascii=False), encoding="utf-8")
            messagebox.showinfo("Export complete", f"Model evidence saved to:\n{destination}")

    def close(self):
        self.running = False
        if self.camera is not None:
            self.camera.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = StickerDetectionApp(root)
    root.mainloop()
