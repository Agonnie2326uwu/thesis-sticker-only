"""
Benchmark dashboard: YOLO26n vs YOLO11n for PSU vehicle sticker detection.

One window for the whole study, for presenting to the panel:

  Overview                 the three tests and where each one stands
  1  Natural Test Set      accuracy and speed on the 210 test images
  2  Field Test            detection by distance (5-10 m)
  3  Synthetic Stressors   motion blur, occlusion, low light, glare
  4  Final Comparison      which model is better under each condition

Each test page explains the test in plain words, shows the comparison table
and charts, and has a Run button that runs the evaluation script in the
background (no terminal needed). Results are read from research_outputs/,
so saved results appear instantly without running anything again.

Usage (from the project folder, with the venv active):
    python dashboard.py
"""

import csv
import math
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.ticker  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "research_outputs"
FIELD_DIR = OUTPUT_DIR / "field_results"
SYNTHETIC_DIR = OUTPUT_DIR / "synthetic_results"
REPORT_PDF = OUTPUT_DIR / "benchmark_report.pdf"
ARCHIVE_DIR = OUTPUT_DIR / "_archive"

# What "Start over" moves into the archive. Field-test captures
# (research_outputs/field_test) are raw data and are never touched.
RESULT_PATHS = [
    OUTPUT_DIR / "test_metrics.csv",
    OUTPUT_DIR / "test_speed.csv",
    OUTPUT_DIR / "test_evaluation.jsonl",
    OUTPUT_DIR / "benchmark_report.pdf",
    FIELD_DIR,
    SYNTHETIC_DIR,
    BASE_DIR / "runs" / "test_eval",
    BASE_DIR / "runs" / "field_eval",
]


def archive_results(paths=None, archive_dir=None):
    """
    Move every existing result into research_outputs/_archive/<date-time>/,
    keeping the folder layout. Nothing is deleted. Returns the archive folder
    (or None when there was nothing to move).
    """
    import shutil

    paths = RESULT_PATHS if paths is None else paths
    archive_dir = ARCHIVE_DIR if archive_dir is None else archive_dir
    existing = [p for p in paths if Path(p).exists()]
    if not existing:
        return None

    target = Path(archive_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in existing:
        path = Path(path)
        try:
            relative = path.relative_to(BASE_DIR)
        except ValueError:
            relative = Path(path.name)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))
    return target

MODELS = ("YOLO26n", "YOLO11n")

# Validated categorical pair (light surface): passes CVD and contrast checks.
COLOURS = {"YOLO26n": "#2a78d6", "YOLO11n": "#eb6834"}
SURFACE = "#fcfcfb"
PANEL = "#f3f2ef"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
ACCENT = "#2a78d6"

STRESSORS = {
    "motion_blur": {"name": "Motion blur", "unit": "blur length L (px)",
                    "fmt": lambda v: f"{v:.0f} px"},
    "occlusion": {"name": "Occlusion", "unit": "share of sticker covered",
                  "fmt": lambda v: f"{v * 100:.0f}% covered"},
    "low_light": {"name": "Low light", "unit": "darkening (gamma)",
                  "fmt": lambda v: f"gamma {v:.2f}"},
    "glare": {"name": "Glare", "unit": "brightening (gain)",
              "fmt": lambda v: f"gain {v:.2f}x"},
}

SCRIPTS = {
    "test": "evaluate_test.py",
    "field": "evaluate_field.py",
    "synthetic": "evaluate_synthetic.py",
}


# ============================================================
# READING RESULTS
# ============================================================

def read_csv(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def modified(path):
    path = Path(path)
    if not path.is_file():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).strftime(
        "%b %d, %Y %I:%M %p")


def by_model(rows):
    return {row.get("model"): row for row in rows if row.get("model")}


def load_test1(output_dir=OUTPUT_DIR):
    metrics = by_model(read_csv(output_dir / "test_metrics.csv"))
    speed = read_csv(output_dir / "test_speed.csv")
    return {
        "metrics": {m: metrics[m] for m in MODELS if m in metrics},
        "speed": [r for r in speed if r.get("model") in MODELS],
        "updated": modified(output_dir / "test_metrics.csv"),
    }


def load_field(field_dir=FIELD_DIR):
    outcomes = read_csv(field_dir / "field_outcomes.csv")
    parked = [r for r in outcomes if r.get("motion", "parked")
              in ("", "parked") and r.get("distance_m")]
    rates = {}
    for model in MODELS:
        by_distance = {}
        for row in parked:
            d = num(row["distance_m"])
            hit = num(row.get(f"{model}_detected"))
            if d is not None and hit is not None:
                by_distance.setdefault(d, []).append(hit)
        rates[model] = {d: sum(v) / len(v) for d, v in sorted(by_distance.items())}
    counts = {}
    for row in parked:
        d = num(row["distance_m"])
        counts[d] = counts.get(d, 0) + 1

    return {
        "metrics": read_csv(field_dir / "field_metrics.csv"),
        "points": by_model(read_csv(field_dir / "field_breaking_points.csv")),
        "comparison": (read_csv(field_dir / "field_comparison.csv") or [{}])[0],
        "rates": rates,
        "counts": dict(sorted(counts.items())),
        "updated": modified(field_dir / "field_breaking_points.csv"),
    }


def load_synthetic(synthetic_dir=SYNTHETIC_DIR):
    steps = {}
    for row in read_csv(synthetic_dir / "synthetic_steps.csv"):
        key = (row["stressor"], row["model"])
        steps.setdefault(key, []).append({
            "x": num(row["step_value"]),
            "rate": num(row["detection_rate"]),
            "map50_95": num(row.get("map50_95")),
            "images": row.get("images"),
        })
    points = {(r["stressor"], r["model"]): r
              for r in read_csv(synthetic_dir / "synthetic_breaking_points.csv")}
    comparison = {r["stressor"]: r for r in
                  read_csv(synthetic_dir / "synthetic_comparison.csv")}
    baseline = read_csv(synthetic_dir / "baseline_images.csv")
    examples = {s: synthetic_dir / "examples" / f"{s}.jpg" for s in STRESSORS
                if (synthetic_dir / "examples" / f"{s}.jpg").is_file()}
    return {
        "steps": steps, "points": points, "comparison": comparison,
        "ranking": read_csv(synthetic_dir / "stressor_ranking.csv"),
        "baseline_used": (int(num(next(iter(steps.values()))[0]["images"]) or 0)
                          if steps else 0),
        "baseline_total": sum(1 for r in baseline if r.get("baseline") == "1"),
        "test_with_sticker": len(baseline),
        "examples": examples,
        "updated": modified(synthetic_dir / "synthetic_breaking_points.csv"),
    }


# ============================================================
# VERDICTS (plain-language winners)
# ============================================================

def higher_better(a, b, tolerance=0.0):
    if a is None or b is None:
        return "—"
    if abs(a - b) <= tolerance:
        return "About the same"
    return MODELS[0] if a > b else MODELS[1]


def stats_verdict(verdict, x50_26, x50_11):
    """Winner for a breaking point, honest about statistical clarity."""
    verdict = verdict or ""
    if verdict.startswith("neither model"):
        return "Both held up"
    if x50_26 is None and x50_11 is not None:
        return f"{MODELS[0]} (never broke)"
    if x50_11 is None and x50_26 is not None:
        return f"{MODELS[1]} (never broke)"
    if x50_26 is None or x50_11 is None:
        return "—"
    leader = MODELS[0] if x50_26 > x50_11 else MODELS[1]
    if verdict.startswith("difference is clear"):
        return leader
    return f"No clear difference ({leader} slightly ahead)"


def scorecard(test1, field, synthetic):
    """Rows for the Final Comparison table: (measure, 26n, 11n, better)."""
    rows = []
    m = test1["metrics"]
    if len(m) == 2:
        for key, label in (("map50_95", "Accuracy: mAP@0.5:0.95"),
                           ("map50", "Accuracy: mAP@0.5"),
                           ("precision", "Precision"),
                           ("recall", "Recall")):
            a, b = num(m["YOLO26n"].get(key)), num(m["YOLO11n"].get(key))
            rows.append((label, f"{a:.3f}", f"{b:.3f}", higher_better(a, b)))

    for device in ("GPU", "CPU"):
        speed = {r["model"]: r for r in test1["speed"]
                 if r.get("device", "").startswith(device)}
        if len(speed) == 2:
            a = num(speed["YOLO26n"].get("fps_runs_mean") or
                    speed["YOLO26n"].get("fps"))
            b = num(speed["YOLO11n"].get("fps_runs_mean") or
                    speed["YOLO11n"].get("fps"))
            sd = max(num(speed["YOLO26n"].get("fps_runs_sd")) or 0,
                     num(speed["YOLO11n"].get("fps_runs_sd")) or 0)
            rows.append((f"Speed on {device} (FPS)", f"{a:.1f}", f"{b:.1f}",
                         higher_better(a, b, tolerance=sd)))

    if len(field["points"]) == 2:
        a = num(field["points"]["YOLO26n"].get("x50_metres"))
        b = num(field["points"]["YOLO11n"].get("x50_metres"))
        rows.append(("Distance: still detected up to (50%)",
                     "—" if a is None else f"{a:.1f} m",
                     "—" if b is None else f"{b:.1f} m",
                     stats_verdict(field["comparison"].get("bootstrap_verdict"),
                                   a, b)))

    for key, info in STRESSORS.items():
        p26 = synthetic["points"].get((key, "YOLO26n"))
        p11 = synthetic["points"].get((key, "YOLO11n"))
        if not p26 or not p11:
            continue
        a, b = num(p26.get("x50")), num(p11.get("x50"))
        verdict = synthetic["comparison"].get(key, {}).get("bootstrap_verdict")
        reached = [x for x in (p26, p11)
                   if not str(x.get("note", "")).startswith("detected every")]
        rows.append((f"{info['name']}: breaks at (50%)",
                     "not reached" if a is None else info["fmt"](a),
                     "not reached" if b is None else info["fmt"](b),
                     stats_verdict(verdict, a, b) if reached else
                     "Both held up"))
    return rows


def most_limiting(ranking):
    """{model: stressor name ranked 1}."""
    result = {}
    for row in ranking:
        if str(row.get("rank_most_limiting")) == "1":
            stressor = row["stressor"]
            name = (STRESSORS[stressor]["name"] if stressor in STRESSORS
                    else stressor.capitalize())
            result[row["model"]] = name
    return result


# ============================================================
# CHARTS
# ============================================================

def new_figure(width=9.6, height=3.6):
    figure = Figure(figsize=(width, height), dpi=100, facecolor=SURFACE)
    return figure


def style(ax, ylabel="", xlabel=""):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)


def title(ax, text):
    ax.set_title(text, loc="left", color=TEXT, fontsize=11, pad=10)


def legend(ax, **kwargs):
    kwargs.setdefault("fontsize", 9)
    ax.legend(frameon=False, labelcolor=TEXT, **kwargs)


def grouped_bars(ax, labels, values_by_model, ymax=None, fmt="{:.3f}",
                 annotate=True):
    width = 0.36
    for i, model in enumerate(MODELS):
        offset = (i - 0.5) * (width + 0.03)
        values = values_by_model[model]
        bars = ax.bar([x + offset for x in range(len(labels))], values,
                      width=width, color=COLOURS[model], label=model,
                      edgecolor=SURFACE, linewidth=1)
        for bar, value in zip(bars, values if annotate else []):
            ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2,
                                            bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    if ymax:
        ax.set_ylim(0, ymax)


def fig_accuracy(test1):
    figure = new_figure(9.6, 3.6)
    ax = figure.add_subplot(1, 2, 1)
    keys = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
            ("map50", "mAP\n@0.5"), ("map50_95", "mAP\n@0.5:0.95")]
    grouped_bars(ax, [k[1] for k in keys],
                 {m: [num(test1["metrics"][m][k[0]]) or 0 for k in keys]
                  for m in MODELS}, ymax=1.2, annotate=False)
    style(ax, "score (0 to 1, higher is better)")
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    title(ax, "Accuracy on the 210 test images")
    legend(ax, loc="upper right", ncol=2)

    ax = figure.add_subplot(1, 2, 2)
    keys = [("tp_at_app_conf", "Found\n(correct)"),
            ("fn_at_app_conf", "Missed"),
            ("fp_at_app_conf", "False\nalarms")]
    values = {m: [num(test1["metrics"][m][k[0]]) or 0 for k in keys]
              for m in MODELS}
    top = max(max(v) for v in values.values())
    grouped_bars(ax, [k[1] for k in keys], values, ymax=top * 1.15,
                 fmt="{:.0f}")
    style(ax, "number of stickers (out of "
              f"{test1['metrics']['YOLO26n'].get('labelled_stickers', '238')})")
    title(ax, "Detections at confidence 0.25")
    figure.tight_layout()
    return figure


def fig_speed(test1):
    figure = new_figure(9.6, 3.2)
    parts = [("preprocess_ms_mean", "Prepare image", 0.35),
             ("inference_ms_mean", "Model", 1.0),
             ("postprocess_ms_mean", "Cleanup (NMS)", 0.6)]
    devices = [d for d in ("GPU", "CPU")
               if any(r.get("device", "").startswith(d) for r in test1["speed"])]
    for index, device in enumerate(devices, start=1):
        ax = figure.add_subplot(1, len(devices), index)
        rows = {r["model"]: r for r in test1["speed"]
                if r.get("device", "").startswith(device)}
        y = list(range(len(MODELS)))[::-1]
        for model, yy in zip(MODELS, y):
            row = rows.get(model)
            if not row:
                continue
            left = 0
            for key, label, alpha in parts:
                value = num(row.get(key)) or 0
                ax.barh(yy, value, left=left, height=0.5,
                        color=COLOURS[model], alpha=alpha, edgecolor=SURFACE,
                        linewidth=2)
                left += value
            fps = num(row.get("fps_runs_mean") or row.get("fps"))
            ax.annotate(f"{left:.1f} ms  ({fps:.1f} FPS)", (left, yy),
                        xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=TEXT)
        ax.set_yticks(y)
        ax.set_yticklabels(MODELS)
        style(ax, xlabel="milliseconds per image (lower is faster)")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_xlim(0, max((num(r.get("total_ms_mean")) or 0)
                           for r in rows.values()) * 1.6)
        title(ax, f"Time per image on the {device}")
    figure.text(0.01, 0.01, "Bar shades, light to dark to medium: prepare "
                "image, model, cleanup (NMS). YOLO26n's cleanup is shorter "
                "because it is NMS-free.", fontsize=8, color=MUTED)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return figure


def logistic(b0, b1, x):
    return 1 / (1 + math.exp(-max(-35, min(35, b0 + b1 * x))))


def fig_distance(field):
    figure = new_figure(9.6, 3.8)
    ax = figure.add_subplot(1, 2, 1)
    all_d = sorted({d for rates in field["rates"].values() for d in rates})
    for model in MODELS:
        rates = field["rates"].get(model, {})
        ax.plot(list(rates), list(rates.values()), "o", color=COLOURS[model],
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                label=f"{model} measured", zorder=3)
        point = field["points"].get(model, {})
        b0, b1 = num(point.get("b0_m")), num(point.get("b1_m"))
        if b0 is not None and b1 is not None and all_d:
            xs = [all_d[0] + i * (all_d[-1] - all_d[0]) / 100 for i in range(101)]
            ax.plot(xs, [logistic(b0, b1, x) for x in xs], "-", linewidth=2,
                    color=COLOURS[model], label=f"{model} fitted")
    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
    ax.axhline(0.9, color=MUTED, linestyle=":", linewidth=1)
    if all_d:
        ax.annotate("breaking point 50%", (all_d[-1], 0.5), xytext=(0, 4),
                    textcoords="offset points", ha="right", fontsize=8,
                    color=MUTED)
        ax.annotate("reliable 90%", (all_d[-1], 0.9), xytext=(0, 4),
                    textcoords="offset points", ha="right", fontsize=8,
                    color=MUTED)
    ax.set_ylim(0, 1.05)
    style(ax, "share of stickers detected", "distance from camera (m)")
    title(ax, "Detection rate by distance")
    legend(ax, loc="lower left", fontsize=8)

    ax = figure.add_subplot(1, 2, 2)
    for model in MODELS:
        rows = [r for r in field["metrics"] if r.get("group") == "distance_m"
                and r.get("model") == model and num(r.get("map50_95")) is not None]
        rows.sort(key=lambda r: num(r["level"]) or 0)
        ax.plot([num(r["level"]) for r in rows],
                [num(r["map50_95"]) for r in rows], "o-", color=COLOURS[model],
                linewidth=2, markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=2, label=model)
    ax.set_ylim(0, 1.05)
    style(ax, "mAP@0.5:0.95", "distance from camera (m)")
    title(ax, "Accuracy (mAP@0.5:0.95) by distance")
    legend(ax, loc="lower left")
    figure.tight_layout()
    return figure


def fig_synthetic(synthetic, metric="rate"):
    figure = new_figure(9.6, 6.4)
    present = [s for s in STRESSORS
               if any((s, m) in synthetic["steps"] for m in MODELS)]
    for index, key in enumerate(present, start=1):
        ax = figure.add_subplot(2, 2, index)
        info = STRESSORS[key]
        for model in MODELS:
            steps = synthetic["steps"].get((key, model), [])
            xs = [s["x"] for s in steps]
            ys = [s["rate"] if metric == "rate" else s["map50_95"] for s in steps]
            ys = [math.nan if y is None else y for y in ys]
            if metric == "rate":
                ax.plot(xs, ys, "o", color=COLOURS[model], markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.5,
                        label=f"{model}", zorder=3)
                point = synthetic["points"].get((key, model), {})
                b0, b1 = num(point.get("b0")), num(point.get("b1"))
                if b0 is not None and b1 is not None and xs:
                    grid = [xs[0] + i * (xs[-1] - xs[0]) / 100 for i in range(101)]
                    ax.plot(grid, [logistic(b0, b1, x) for x in grid], "-",
                            color=COLOURS[model], linewidth=2)
            else:
                ax.plot(xs, ys, "o-", color=COLOURS[model], linewidth=2,
                        markersize=5, markeredgecolor=SURFACE, label=model)
        if metric == "rate":
            ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
            ax.axhline(0.9, color=MUTED, linestyle=":", linewidth=1)
        ax.set_ylim(0, 1.05)
        if key == "occlusion":
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
        style(ax, "share detected" if metric == "rate" else "mAP@0.5:0.95",
              info["unit"])
        title(ax, info["name"])
        if index == 1:
            legend(ax, loc="lower left")
    figure.text(0.01, 0.005,
                ("Dots = measured, lines = fitted curve. Dashed line = breaking "
                 "point (50% detected); dotted line = reliable detection (90%). "
                 "Further right = harder.") if metric == "rate" else
                "Further right = harder.", fontsize=8, color=MUTED)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    return figure


def fig_ranking(ranking):
    rows = [r for r in ranking if r.get("model") in MODELS]
    stressors = []
    for r in rows:
        if r["stressor"] not in stressors:
            stressors.append(r["stressor"])
    names = [STRESSORS[s]["name"] if s in STRESSORS else s.capitalize()
             for s in stressors]
    figure = new_figure(9.6, 0.9 + 0.55 * max(1, len(stressors)))
    ax = figure.add_subplot(1, 1, 1)
    height = 0.36
    for i, model in enumerate(MODELS):
        for j, stressor in enumerate(stressors):
            row = next((r for r in rows if r["model"] == model
                        and r["stressor"] == stressor), None)
            if not row:
                continue
            value = num(row.get("range_fraction"))
            not_reached = row.get("range_fraction") == "not reached"
            width = 1.0 if not_reached else max(0.0, min(1.0, value or 0))
            y = j + (i - 0.5) * (height + 0.04)
            ax.barh(y, width, height=height, color=COLOURS[model],
                    alpha=0.35 if not_reached else 1.0,
                    hatch="//" if not_reached else None,
                    edgecolor=SURFACE, linewidth=1,
                    label=model if j == 0 else None)
            ax.annotate("never broke" if not_reached else f"{width:.0%}",
                        (width, y), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8, color=TEXT)
    ax.set_yticks(range(len(stressors)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.18)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    style(ax, xlabel="how far into the tested range the model still detects "
                     "50% of stickers (longer bar = more tolerant)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    title(ax, "How much of each stressor each model tolerates")
    legend(ax, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    figure.tight_layout()
    return figure


# ============================================================
# THE WINDOW
# ============================================================

class Dashboard:
    PAGES = ("Overview", "1  Natural Test Set", "2  Field Test (Distance)",
             "3  Synthetic Stressors", "4  Final Comparison")

    def __init__(self, root):
        self.root = root
        self.root.title("YOLO26n vs YOLO11n - Sticker Detection Benchmark")
        self.root.geometry("1280x820")
        self.root.minsize(1000, 640)
        self.root.configure(bg=SURFACE)
        self.process = None
        self.lines = queue.Queue()
        self.log_lines = []
        self.figures = []
        self.images = []
        self.current = None

        self._style()
        self._layout()
        self.show(self.PAGES[0])
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # ---------------- layout ----------------

    def _style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        base = ("Segoe UI", 10)
        style.configure(".", background=SURFACE, foreground=TEXT, font=base)
        style.configure("Side.TFrame", background=PANEL)
        style.configure("Side.TButton", background=PANEL, foreground=TEXT,
                        anchor="w", padding=(14, 10), relief="flat",
                        font=("Segoe UI", 10))
        style.map("Side.TButton", background=[("active", "#e7e5e0")])
        style.configure("Current.TButton", background="#ffffff",
                        foreground=ACCENT, anchor="w", padding=(14, 10),
                        relief="flat", font=("Segoe UI", 10, "bold"))
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("H2.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Card.TFrame", background="#ffffff", relief="solid",
                        borderwidth=1)
        style.configure("CardValue.TLabel", background="#ffffff",
                        font=("Segoe UI", 16, "bold"))
        style.configure("CardLabel.TLabel", background="#ffffff",
                        foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Run.TButton", font=("Segoe UI", 10, "bold"),
                        padding=(14, 6))
        style.configure("Treeview", rowheight=26, font=base,
                        background="#ffffff", fieldbackground="#ffffff")
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

    def _layout(self):
        header = tk.Frame(self.root, bg=TEXT, height=52)
        header.pack(fill="x")
        tk.Label(header, text="PSU Vehicle Sticker Detection  ·  YOLO26n vs "
                 "YOLO11n Benchmark", bg=TEXT, fg="#ffffff",
                 font=("Segoe UI", 14, "bold")).pack(side="left", padx=18,
                                                     pady=12)
        legend_frame = tk.Frame(header, bg=TEXT)
        legend_frame.pack(side="right", padx=18)
        for model in MODELS:
            tk.Label(legend_frame, text="■", fg=COLOURS[model], bg=TEXT,
                     font=("Segoe UI", 14)).pack(side="left")
            tk.Label(legend_frame, text=model + "   ", fg="#ffffff", bg=TEXT,
                     font=("Segoe UI", 10)).pack(side="left")

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True)
        self.body = body

        side = ttk.Frame(body, style="Side.TFrame", width=240)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        ttk.Label(side, text="STEPS", background=PANEL, foreground=MUTED,
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=16,
                                                     pady=(16, 6))
        self.nav = {}
        for page in self.PAGES:
            button = ttk.Button(side, text=page, style="Side.TButton",
                                command=lambda p=page: self.show(p))
            button.pack(fill="x")
            self.nav[page] = button

        ttk.Separator(side).pack(fill="x", pady=12)
        ttk.Button(side, text="Export PDF report", style="Side.TButton",
                   command=self.export_pdf).pack(fill="x")
        ttk.Button(side, text="Open live demo (camera app)",
                   style="Side.TButton", command=self.open_app).pack(fill="x")
        ttk.Button(side, text="Refresh results", style="Side.TButton",
                   command=lambda: self.show(self.current)).pack(fill="x")
        ttk.Separator(side).pack(fill="x", pady=12)
        ttk.Button(side, text="Start over (archive results)",
                   style="Side.TButton", command=self.start_over).pack(fill="x")

        main = ttk.Frame(body)
        main.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(main, bg=SURFACE, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main, orient="vertical",
                                  command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.page = ttk.Frame(self.canvas, padding=(28, 20))
        self.page_window = self.canvas.create_window((0, 0), window=self.page,
                                                     anchor="nw")
        self.page.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(
            self.page_window, width=e.width))
        self.root.bind_all("<MouseWheel>", self._wheel)
        self.root.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-3, "units"))
        self.root.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(3, "units"))

        # Run panel (hidden until a test runs)
        self.run_panel = ttk.Frame(self.root, padding=(16, 8))
        self.run_label = ttk.Label(self.run_panel, text="", style="Muted.TLabel")
        self.run_label.pack(side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(self.run_panel, mode="indeterminate",
                                        length=220)
        self.progress.pack(side="left", padx=8)
        ttk.Button(self.run_panel, text="Show details",
                   command=self.show_log).pack(side="left", padx=4)
        self.stop_button = ttk.Button(self.run_panel, text="Stop",
                                      command=self.stop)
        self.stop_button.pack(side="left")

    def _wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120) * 3, "units")

    # ---------------- page helpers ----------------

    def show(self, page):
        self.current = page
        for name, button in self.nav.items():
            button.configure(style="Current.TButton" if name == page
                             else "Side.TButton")
        for child in self.page.winfo_children():
            child.destroy()
        for figure in self.figures:
            figure.clear()
        self.figures, self.images = [], []
        self.canvas.yview_moveto(0)
        builder = {
            self.PAGES[0]: self.page_overview,
            self.PAGES[1]: self.page_test1,
            self.PAGES[2]: self.page_field,
            self.PAGES[3]: self.page_synthetic,
            self.PAGES[4]: self.page_final,
        }[page]
        try:
            builder()
        except Exception as exc:     # a damaged CSV must not crash the show
            self.text(f"Could not show these results: {exc}", muted=True)

    def heading(self, text, subtitle=None):
        ttk.Label(self.page, text=text, style="Title.TLabel").pack(anchor="w")
        if subtitle:
            ttk.Label(self.page, text=subtitle, style="Muted.TLabel",
                      wraplength=900, justify="left").pack(anchor="w",
                                                           pady=(2, 10))

    def section(self, text):
        ttk.Label(self.page, text=text, style="H2.TLabel").pack(anchor="w",
                                                                pady=(18, 6))

    def text(self, value, muted=False):
        ttk.Label(self.page, text=value, wraplength=900, justify="left",
                  style="Muted.TLabel" if muted else "TLabel").pack(
            anchor="w", pady=2)

    def steps_box(self, steps):
        box = ttk.Frame(self.page)
        box.pack(anchor="w", fill="x", pady=(0, 4))
        for index, step in enumerate(steps, start=1):
            row = ttk.Frame(box)
            row.pack(anchor="w", fill="x", pady=1)
            tk.Label(row, text=str(index), bg=ACCENT, fg="#ffffff", width=2,
                     font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 8))
            ttk.Label(row, text=step, wraplength=860,
                      justify="left").pack(side="left", anchor="w")

    def cards(self, items):
        row = ttk.Frame(self.page)
        row.pack(fill="x", pady=(6, 4))
        for label, value, note in items:
            card = ttk.Frame(row, style="Card.TFrame", padding=(14, 10))
            card.pack(side="left", padx=(0, 10), fill="y")
            ttk.Label(card, text=label, style="CardLabel.TLabel").pack(anchor="w")
            ttk.Label(card, text=value, style="CardValue.TLabel").pack(anchor="w")
            if note:
                ttk.Label(card, text=note, style="CardLabel.TLabel",
                          wraplength=200, justify="left").pack(anchor="w")

    def table(self, columns, rows, widths=None, winner_column=None):
        frame = ttk.Frame(self.page)
        frame.pack(fill="x", pady=(4, 6))
        tree = ttk.Treeview(frame, columns=columns, show="headings",
                            height=len(rows))
        for i, column in enumerate(columns):
            tree.heading(column, text=column)
            tree.column(column, width=(widths or [180] * len(columns))[i],
                        anchor="w" if i == 0 else "center")
        tree.tag_configure("win", font=("Segoe UI", 10, "bold"))
        for row in rows:
            tag = ()
            if (winner_column is not None
                    and str(row[winner_column]).startswith(MODELS)):
                tag = ("win",)
            tree.insert("", "end", values=row, tags=tag)
        tree.pack(fill="x")

    def figure(self, figure):
        self.figures.append(figure)
        canvas = FigureCanvasTkAgg(figure, master=self.page)
        canvas.draw()
        canvas.get_tk_widget().pack(anchor="w", pady=(6, 6))

    def image(self, path, caption):
        try:
            from PIL import Image, ImageTk
        except ImportError:
            return
        with Image.open(path) as source:
            picture = source.copy()
        picture.thumbnail((900, 200))
        photo = ImageTk.PhotoImage(picture)
        self.images.append(photo)
        ttk.Label(self.page, text=caption, style="Muted.TLabel").pack(
            anchor="w", pady=(6, 0))
        ttk.Label(self.page, image=photo).pack(anchor="w")

    def run_bar(self, key, label, extra=None, note=None):
        row = ttk.Frame(self.page)
        row.pack(anchor="w", pady=(4, 8))
        ttk.Button(row, text=label, style="Run.TButton",
                   command=lambda: self.run(key, extra or [])).pack(side="left")
        if note:
            ttk.Label(row, text="   " + note, style="Muted.TLabel").pack(
                side="left")

    def no_results(self, what):
        box = ttk.Frame(self.page, style="Card.TFrame", padding=16)
        box.pack(fill="x", pady=8)
        ttk.Label(box, text="No results yet", style="CardValue.TLabel").pack(
            anchor="w")
        ttk.Label(box, text=what, style="CardLabel.TLabel", wraplength=850,
                  justify="left").pack(anchor="w")

    # ---------------- pages ----------------

    def page_overview(self):
        test1, field, synthetic = load_test1(), load_field(), load_synthetic()
        self.heading(
            "Which detects PSU vehicle stickers better: YOLO26n or YOLO11n?",
            "Both models were trained the same way on 1,400 phone photos taken "
            "at the PSU main gate (980 for training, 210 for validation, 210 "
            "for the final test). They are then compared in three tests, from "
            "normal conditions to increasingly difficult ones.")

        def status(updated):
            return (f"Done ({updated.split(' ')[0]} {updated.split(' ')[1]})"
                    if updated else "Not run yet")

        self.section("The three tests")
        self.table(
            ("Test", "What it checks", "Question it answers", "Status"),
            [("1  Natural test set", "210 photos the models never saw",
              "Which model is more accurate and faster?",
              status(test1["updated"])),
             ("2  Field test", "Webcam at the gate, 5 to 10 m",
              "Up to what distance is the sticker still found?",
              status(field["updated"])),
             ("3  Synthetic stressors", "Blurred, covered, darker, brighter",
              "At what level does each model fail?",
              status(synthetic["updated"]))],
            widths=[180, 250, 330, 180])

        self.section("How a detection is counted")
        self.text("A sticker counts as detected when the model draws a box with "
                  "a confidence of at least 0.25 that overlaps the real sticker "
                  "box by at least 50% (IoU ≥ 0.5). The breaking point is the "
                  "level at which a model finds only half of the stickers; "
                  "the reliable point is where it still finds 90%.")

        rows = scorecard(test1, field, synthetic)
        if rows:
            self.section("Results so far")
            self.table(("Measure", "YOLO26n", "YOLO11n", "Better"),
                       rows[:6], widths=[300, 160, 160, 320], winner_column=3)
            self.text("See Final Comparison for the full scorecard.", muted=True)

    def page_test1(self):
        test1 = load_test1()
        self.heading(
            "Test 1 · Natural test set",
            "Both models take the same final exam: 210 photos (238 stickers) "
            "they never saw during training. Each model's boxes are compared "
            "with the correct boxes, and the time per photo is measured.")
        self.steps_box([
            "Load the 210 test photos and their correct sticker boxes.",
            "Each model finds stickers in every photo.",
            "Compare with the correct boxes: precision, recall, mAP.",
            "Time each model photo by photo: 3 runs on the GPU and 3 on the "
            "CPU, after a full warm-up pass.",
        ])
        self.run_bar("test", "Run Test 1", note="about 10-20 minutes "
                     "(GTX 1660 SUPER desktop)")

        if len(test1["metrics"]) < 2:
            self.no_results("Click Run Test 1 to create the results.")
            return

        m = test1["metrics"]
        a, b = num(m["YOLO26n"]["map50_95"]), num(m["YOLO11n"]["map50_95"])
        gpu = {r["model"]: r for r in test1["speed"]
               if r.get("device", "").startswith("GPU")}
        cards = [("Most accurate (mAP@0.5:0.95)",
                  higher_better(a, b), f"{a:.3f} vs {b:.3f}")]
        if len(gpu) == 2:
            fa = num(gpu["YOLO26n"].get("fps_runs_mean") or gpu["YOLO26n"].get("fps"))
            fb = num(gpu["YOLO11n"].get("fps_runs_mean") or gpu["YOLO11n"].get("fps"))
            cards.append(("Fastest on GPU", higher_better(fa, fb),
                          f"{fa:.1f} vs {fb:.1f} FPS"))
        cards.append(("Stickers found (of 238)",
                      f"{m['YOLO26n']['tp_at_app_conf']} vs "
                      f"{m['YOLO11n']['tp_at_app_conf']}",
                      "YOLO26n vs YOLO11n, at confidence 0.25"))
        self.cards(cards)
        self.text(f"Results from {test1['updated']}.", muted=True)

        self.section("Accuracy")
        self.figure(fig_accuracy(test1))
        rows = []
        for key, label in (("precision", "Precision"), ("recall", "Recall"),
                           ("f1", "F1"), ("map50", "mAP@0.5"),
                           ("map50_95", "mAP@0.5:0.95")):
            x, y = num(m["YOLO26n"][key]), num(m["YOLO11n"][key])
            rows.append((label, f"{x:.3f}", f"{y:.3f}", higher_better(x, y)))
        for key, label in (("tp_at_app_conf", "Found (TP)"),
                           ("fn_at_app_conf", "Missed (FN)"),
                           ("fp_at_app_conf", "False alarms (FP)")):
            x, y = num(m["YOLO26n"][key]), num(m["YOLO11n"][key])
            better = (higher_better(x, y) if key == "tp_at_app_conf" else
                      higher_better(-x, -y))
            rows.append((label, f"{x:.0f}", f"{y:.0f}", better))
        self.table(("Measure", "YOLO26n", "YOLO11n", "Better"), rows,
                   widths=[260, 160, 160, 260], winner_column=3)

        if test1["speed"]:
            self.section("Speed")
            self.figure(fig_speed(test1))
            rows = []
            for r in test1["speed"]:
                fps = num(r.get("fps_runs_mean") or r.get("fps"))
                sd = num(r.get("fps_runs_sd"))
                rows.append((r["model"], r["device"].split(" (")[0],
                             f"{num(r['preprocess_ms_mean']):.2f}",
                             f"{num(r['inference_ms_mean']):.2f}",
                             f"{num(r['postprocess_ms_mean']):.2f}",
                             f"{num(r['total_ms_mean']):.2f}",
                             f"{fps:.1f}" + (f" ± {sd:.1f}" if sd else "")))
            self.table(("Model", "Device", "Prepare (ms)", "Model (ms)",
                        "Cleanup (ms)", "Total (ms)", "FPS (mean ± SD)"),
                       rows, widths=[110, 80, 110, 110, 110, 110, 150])

    def page_field(self):
        field = load_field()
        self.heading(
            "Test 2 · Field test (distance)",
            "The training photos were all taken at 5 m or closer. This test "
            "checks how far away each model can still find the sticker, using "
            "new webcam photos at the PSU gate at 5 m (baseline) and 6, 7, 8, "
            "9 and 10 m, in daylight, with the vehicle parked at each mark.")
        self.steps_box([
            "Capture photos with the camera app's Field Test tab at each "
            "distance (RTX 2050 laptop at the gate).",
            "Label the stickers on the Ultralytics Platform "
            "(dataset: student-vehicle-sticker-field).",
            "Each model finds stickers in every field photo.",
            "Fit a curve of detection rate vs. distance to find where each "
            "model drops to 50% (breaking point) and 90% (reliable).",
        ])
        self.run_bar("field", "Run Test 2", note="run on the RTX 2050 laptop "
                     "after labelling the field photos")

        if len(field["points"]) < 2:
            self.no_results("The field test has not been evaluated yet. After "
                            "the field photos are captured and labelled, click "
                            "Run Test 2 on the laptop. To show the results on "
                            "this computer, copy the laptop's "
                            "research_outputs/field_results folder here.")
            return

        p = field["points"]
        comparison = field["comparison"]
        a, b = num(p["YOLO26n"].get("x50_metres")), num(p["YOLO11n"].get("x50_metres"))
        self.cards([
            ("YOLO26n still finds half the stickers up to",
             "—" if a is None else f"{a:.1f} m", p["YOLO26n"].get("note_metres", "")),
            ("YOLO11n still finds half the stickers up to",
             "—" if b is None else f"{b:.1f} m", p["YOLO11n"].get("note_metres", "")),
            ("Holds up better at distance",
             stats_verdict(comparison.get("bootstrap_verdict"), a, b),
             f"McNemar p = {num(comparison.get('mcnemar_exact_p')) or 0:.3f}"),
        ])
        self.text(f"Results from {field['updated']}.", muted=True)

        self.section("Detection by distance")
        self.figure(fig_distance(field))

        rows = []
        for model in MODELS:
            q = p[model]

            def show(v, unit, digits=1):
                v = num(v)
                return "—" if v is None else f"{v:.{digits}f} {unit}"
            rows.append((model, show(q.get("x50_metres"), "m"),
                         show(q.get("x90_metres"), "m"),
                         show(q.get("x50_px_width"), "px"),
                         show(q.get("x90_px_width"), "px"),
                         q.get("method_metres") or q.get("method_px") or ""))
        self.table(("Model", "Breaking point (50%)", "Reliable (90%)",
                    "Sticker width at 50%", "Sticker width at 90%", "Method"),
                   rows, widths=[110, 170, 150, 170, 170, 260])
        self.text("Sticker width is measured in pixels at the 640-pixel model "
                  "input, so other researchers can compare with their own "
                  "cameras.", muted=True)

        if field["counts"]:
            self.text("Photos per distance (parked): " + ", ".join(
                f"{d:g} m: {n}" for d, n in field["counts"].items()) + ".",
                muted=True)

    def page_synthetic(self):
        synthetic = load_synthetic()
        self.heading(
            "Test 3 · Synthetic stressors",
            "Real photos mix many conditions, so it is hard to tell which one "
            "caused a miss. Here each condition is tested alone: the test "
            "photos that both models already detect are made gradually "
            "blurrier, more covered, darker or brighter, and both models are "
            "checked at every step.")
        self.steps_box([
            "Pick the baseline photos: test photos where both models find "
            "every sticker.",
            "Make 65 copies of each: 21 blur steps, 10 covering steps, 17 "
            "darker and 17 brighter steps.",
            "Each model finds stickers in every copy.",
            "Fit a curve per condition to find the breaking point (50%) and "
            "reliable point (90%).",
        ])
        row = ttk.Frame(self.page)
        row.pack(anchor="w", pady=(4, 8))
        ttk.Button(row, text="Run Test 3", style="Run.TButton",
                   command=lambda: self.run("synthetic", [])).pack(side="left")
        ttk.Button(row, text="Quick trial (5 photos)",
                   command=lambda: self.run("synthetic", ["--max-images", "5"])
                   ).pack(side="left", padx=8)
        ttk.Label(row, text="full run about 45-60 minutes (GTX 1660 SUPER "
                  "desktop)", style="Muted.TLabel").pack(side="left")

        if not synthetic["points"]:
            self.no_results("Click Run Test 3 to create the results.")
            return

        used, total = synthetic["baseline_used"], synthetic["baseline_total"]
        if used and total and used < total:
            self.text(f"These results are from a trial run with only {used} "
                      f"of {total} baseline photos. Run Test 3 for the real "
                      "results.", muted=False)
        self.cards([
            ("Baseline photos used", f"{used}",
             f"of {synthetic['test_with_sticker']} test photos with a sticker"),
            ("Steps per photo", "65", "blur 21 · covering 10 · dark 17 · "
             "bright 17"),
        ])
        self.text(f"Results from {synthetic['updated']}.", muted=True)

        self.section("What the models were tested on")
        for key, path in synthetic["examples"].items():
            self.image(path, f"{STRESSORS[key]['name']}: one sticker at five "
                             "levels, from none to the hardest")

        self.section("Detection rate at each step")
        self.figure(fig_synthetic(synthetic, "rate"))

        rows = []
        for key, info in STRESSORS.items():
            p26 = synthetic["points"].get((key, "YOLO26n"))
            p11 = synthetic["points"].get((key, "YOLO11n"))
            if not p26 or not p11:
                continue

            def show(p, field_name):
                v = num(p.get(field_name))
                return "not reached" if v is None else info["fmt"](v)
            verdict = synthetic["comparison"].get(key, {}).get("bootstrap_verdict")
            rows.append((info["name"], show(p26, "x50"), show(p11, "x50"),
                         show(p26, "x90"), show(p11, "x90"),
                         stats_verdict(verdict, num(p26.get("x50")),
                                       num(p11.get("x50")))))
        self.table(("Condition", "YOLO26n breaks at", "YOLO11n breaks at",
                    "YOLO26n reliable to", "YOLO11n reliable to",
                    "Tolerates it better"),
                   rows, widths=[120, 150, 150, 150, 150, 300],
                   winner_column=5)
        self.text("\"Breaks at\" = the level where the model finds only 50% of "
                  "stickers; \"reliable to\" = where it still finds 90%. "
                  "\"No clear difference\" means the 95% confidence interval "
                  "of the difference includes zero.", muted=True)

        self.section("Accuracy (mAP@0.5:0.95) at each step")
        self.figure(fig_synthetic(synthetic, "map"))

    def page_final(self):
        test1, field, synthetic = load_test1(), load_field(), load_synthetic()
        self.heading(
            "Final comparison",
            "All results side by side. \"Better\" shows which model wins each "
            "measure; for the stressors it only names a winner when the "
            "difference is statistically clear.")
        rows = scorecard(test1, field, synthetic)
        if not rows:
            self.no_results("Run the tests first.")
            return
        self.table(("Measure", "YOLO26n", "YOLO11n", "Better"), rows,
                   widths=[320, 170, 170, 330], winner_column=3)

        wins = {m: sum(1 for r in rows if r[3].startswith(m)) for m in MODELS}
        limiting = most_limiting(synthetic["ranking"])
        cards = [(f"Measures won by {m}", str(wins[m]), None) for m in MODELS]
        for model in MODELS:
            if model in limiting:
                cards.append((f"Hardest condition for {model}",
                              limiting[model], "reached its breaking point "
                              "earliest"))
        self.cards(cards)

        if synthetic["ranking"]:
            self.section("Which condition limits each model most")
            self.figure(fig_ranking(synthetic["ranking"]))
            if not any(r.get("stressor") == "distance"
                       for r in synthetic["ranking"]):
                self.text("Distance is added here after the field test: run "
                          "Test 2, copy its results to this computer, then "
                          "run Test 3 again.", muted=True)

    # ---------------- running scripts ----------------

    def run(self, key, extra):
        if self.process is not None:
            messagebox.showinfo("Test running",
                                "Wait for the current test to finish.")
            return
        script = BASE_DIR / SCRIPTS[key]
        if not script.is_file():
            messagebox.showerror("Missing script", f"{script.name} was not "
                                 "found in the project folder.")
            return

        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
                   PYTHONUNBUFFERED="1")
        self.log_lines = [f"$ python {script.name} {' '.join(extra)}".strip()]
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(script), *extra], cwd=str(BASE_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        threading.Thread(target=self._reader, daemon=True).start()

        self.run_label.configure(text=f"Running {script.name} ...")
        self.run_panel.pack(side="bottom", fill="x", before=self.body)
        self.progress.start(12)
        self.root.after(200, self._poll, script.name)

    def _reader(self):
        buffer = b""
        stream = self.process.stdout
        while True:
            chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(1)
            if not chunk:
                break
            buffer += chunk
            parts = buffer.replace(b"\r", b"\n").split(b"\n")
            buffer = parts.pop()
            for part in parts:
                text = part.decode("utf-8", errors="replace").strip()
                if text:
                    self.lines.put(text)
        if buffer.strip():
            self.lines.put(buffer.decode("utf-8", errors="replace").strip())

    def _poll(self, name):
        last = None
        while not self.lines.empty():
            last = self.lines.get_nowait()
            if not self.log_lines or self.log_lines[-1] != last:
                self.log_lines.append(last)
        if last:
            self.run_label.configure(text=f"{name}:  {last[:140]}")
        if hasattr(self, "log_text") and self.log_text.winfo_exists():
            self._refresh_log()

        if self.process is not None and self.process.poll() is None:
            self.root.after(300, self._poll, name)
            return

        code = self.process.returncode if self.process else -1
        self.process = None
        self.progress.stop()
        if code == 0:
            self.run_label.configure(text=f"{name} finished. Results updated.")
            self.root.after(4000, self.run_panel.pack_forget)
        else:
            reason = next((l for l in reversed(self.log_lines)
                           if l and not l.startswith("File ")), "")
            self.run_label.configure(text=f"{name} stopped: {reason[:140]}")
            messagebox.showwarning("Test stopped", f"{name} did not finish.\n\n"
                                   f"{reason}\n\nClick 'Show details' for the "
                                   "full output.")
        self.show(self.current)

    def stop(self):
        if self.process is not None and messagebox.askyesno(
                "Stop test", "Stop the running test? Its results will not be "
                "saved."):
            self.process.terminate()

    def show_log(self):
        window = tk.Toplevel(self.root)
        window.title("Test output")
        window.geometry("900x500")
        self.log_text = tk.Text(window, wrap="none", font=("Consolas", 9),
                                bg="#ffffff")
        self.log_text.pack(fill="both", expand=True)
        self._refresh_log()

    def _refresh_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("end", "\n".join(self.log_lines[-2000:]))
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # ---------------- other actions ----------------

    def start_over(self):
        if self.process is not None:
            messagebox.showinfo("Test running",
                                "Wait for the current test to finish.")
            return
        if not messagebox.askyesno(
                "Start over",
                "Move all current results (Test 1, Test 2, Test 3 and the PDF "
                "report) into research_outputs/_archive?\n\nNothing is "
                "deleted, and field-test photos are not touched. Every page "
                "will show 'No results yet' until you run the tests again."):
            return
        try:
            target = archive_results()
        except OSError as exc:
            messagebox.showerror("Could not archive", f"{exc}\n\nClose any "
                                 "result file open in Excel or a PDF viewer, "
                                 "then try again.")
            return
        messagebox.showinfo(
            "Ready to start over",
            "No previous results to archive." if target is None else
            f"Previous results moved to:\n{target}\n\nStart with Test 1.")
        self.show(self.PAGES[1])

    def open_app(self):
        app = BASE_DIR / "app.py"
        if not app.is_file():
            messagebox.showerror("Missing app", "app.py was not found.")
            return
        subprocess.Popen([sys.executable, str(app)], cwd=str(BASE_DIR))

    def export_pdf(self):
        from matplotlib.backends.backend_pdf import PdfPages

        test1, field, synthetic = load_test1(), load_field(), load_synthetic()
        figures = []

        cover = new_figure(8.27, 11.69)
        cover.text(0.08, 0.92, "YOLO26n vs YOLO11n", fontsize=24,
                   color=TEXT, weight="bold")
        cover.text(0.08, 0.885, "PSU Vehicle Sticker Detection Benchmark",
                   fontsize=14, color=MUTED)
        cover.text(0.08, 0.86, datetime.now().strftime("%B %d, %Y"),
                   fontsize=10, color=MUTED)
        rows = scorecard(test1, field, synthetic)
        if rows:
            ax = cover.add_axes((0.08, 0.35, 0.84, 0.45))
            ax.axis("off")
            table = ax.table(cellText=[list(r) for r in rows],
                             colLabels=["Measure", "YOLO26n", "YOLO11n",
                                        "Better"],
                             loc="upper left", cellLoc="left",
                             colWidths=[0.42, 0.17, 0.17, 0.24])
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.6)
            for (r, c), cell in table.get_celld().items():
                cell.set_edgecolor(GRID)
                if r == 0:
                    cell.set_text_props(weight="bold")
                elif c == 3 and cell.get_text().get_text() in MODELS:
                    cell.get_text().set_color(COLOURS[cell.get_text().get_text()])
        figures.append(cover)

        if len(test1["metrics"]) == 2:
            figures.append(fig_accuracy(test1))
            if test1["speed"]:
                figures.append(fig_speed(test1))
        if len(field["points"]) == 2:
            figures.append(fig_distance(field))
        if synthetic["points"]:
            figures.append(fig_synthetic(synthetic, "rate"))
            figures.append(fig_synthetic(synthetic, "map"))
        if synthetic["ranking"]:
            figures.append(fig_ranking(synthetic["ranking"]))

        try:
            REPORT_PDF.parent.mkdir(parents=True, exist_ok=True)
            with PdfPages(REPORT_PDF) as pdf:
                for figure in figures:
                    pdf.savefig(figure, facecolor=SURFACE)
        except PermissionError:
            messagebox.showerror("Export failed", "Close benchmark_report.pdf "
                                 "if it is open, then try again.")
            return
        messagebox.showinfo("PDF saved", f"Saved:\n{REPORT_PDF}")
        try:
            os.startfile(REPORT_PDF)          # Windows
        except AttributeError:
            pass

    def close(self):
        if self.process is not None:
            if not messagebox.askyesno("Test running", "A test is still "
                                       "running. Stop it and close?"):
                return
            self.process.terminate()
        self.root.destroy()


def main():
    root = tk.Tk()
    Dashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()