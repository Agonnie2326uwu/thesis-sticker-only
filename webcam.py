
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

# --------------------------------------------------
# MODEL CONFIGURATION
# Change these paths to your actual trained weights.
# --------------------------------------------------
BASE_DIR = Path(__file__).parent

YOLO26_PATH = (
    BASE_DIR / "runs" / "detect" / "runs"
    / "yolo26n_sticker_split80_10_10"
    / "weights" / "best.pt"
)

YOLO11_PATH = (
    BASE_DIR / "runs" / "detect" / "runs"
    / "yolo11n_sticker_split80_10_10"
    / "weights" / "best.pt"
)

IMGSZ = 640
CONFIDENCE = 0.25
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# VERIFY MODELS
# --------------------------------------------------
for name, path in [
    ("YOLO26n", YOLO26_PATH),
    ("YOLO11n", YOLO11_PATH),
]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{name} weights not found:\n{path}\n"
            "Check the model path and finish training first."
        )

print("Loading YOLO26n...")
model26 = YOLO(str(YOLO26_PATH))

print("Loading YOLO11n...")
model11 = YOLO(str(YOLO11_PATH))

print(f"Using device: {DEVICE}")

# --------------------------------------------------
# OPEN WEBCAM
# --------------------------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Cannot open webcam. Check camera connection or permissions."
    )

print("Both models are running. Press Q to quit.")

try:
    while True:
        ok, frame = cap.read()

        if not ok:
            print("Failed to read webcam frame.")
            break

        # Both models receive the exact same frame.
        result26 = model26.predict(
            source=frame,
            imgsz=IMGSZ,
            conf=CONFIDENCE,
            device=DEVICE,
            verbose=False
        )[0]

        result11 = model11.predict(
            source=frame,
            imgsz=IMGSZ,
            conf=CONFIDENCE,
            device=DEVICE,
            verbose=False
        )[0]

        # Draw each model's detections separately.
        view26 = result26.plot()
        view11 = result11.plot()

        cv2.putText(
            view26, "YOLO26n",
            (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )

        cv2.putText(
            view11, "YOLO11n",
            (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 255), 2
        )

        # Resize both panels to the same height.
        height = 480

        def resize_height(image, target_height):
            h, w = image.shape[:2]
            width = int(w * target_height / h)
            return cv2.resize(image, (width, target_height))

        view26 = resize_height(view26, height)
        view11 = resize_height(view11, height)

        # Side-by-side comparison.
        comparison = cv2.hconcat([view26, view11])

        cv2.imshow(
            "Sticker Detection Comparison: YOLO26n vs YOLO11n",
            comparison
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
