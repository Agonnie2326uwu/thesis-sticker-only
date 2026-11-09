"""
Camera discovery for the sticker-detection app.

OpenCV can only open a camera by its DirectShow index. Windows does not
expose the physical USB port to the application, but DirectShow does expose
the camera names, in the same order as those indexes. We use the names to
tell the external USB camera apart from the laptop's built-in camera and to
open the USB one first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2

try:                            # Windows-only; optional so the app still runs
    from pygrabber.dshow_graph import FilterGraph
except Exception:               # without it (index-based fallback)
    FilterGraph = None

AUTO_CAMERA_LABEL = "Auto (prefer USB)"
MAX_CAMERA_INDEX = 6

# Words used by laptop makers for the built-in camera. Anything else is
# treated as an external (USB) camera.
BUILT_IN_HINTS = (
    "acer", "integrated", "built-in", "built in", "internal", "user facing",
    "front camera", "front cam", "laptop", "notebook", "facetime", "isight",
    "windows hello", "ir camera", "truevision", "easycamera", "thinkpad",
    "ideapad", "surface camera",
)


class CameraError(RuntimeError):
    """No camera could be opened."""


@dataclass(frozen=True)
class CameraInfo:
    index: int
    name: str
    named: bool = True

    @property
    def is_usb(self) -> bool:
        return self.named and not is_built_in_camera(self.name)

    @property
    def label(self) -> str:
        return f"{self.name} (index {self.index})" if self.named else self.name


def is_built_in_camera(name: str) -> bool:
    lower = (name or "").lower()
    return any(hint in lower for hint in BUILT_IN_HINTS)


def camera_names() -> list[str]:
    """DirectShow camera names, in index order; [] if pygrabber is missing."""
    global FilterGraph
    if FilterGraph is None:
        try:
            from pygrabber.dshow_graph import FilterGraph as graph_class
            FilterGraph = graph_class
        except Exception:
            return []
    # The first DirectShow/COM call in a new environment can return nothing
    # while pygrabber builds its cache, so try once more before giving up.
    for attempt in range(2):
        try:
            names = list(FilterGraph().get_input_devices())
        except Exception:
            names = []
        if names:
            return names
        if attempt == 0:
            time.sleep(0.1)
    return []


def camera_choices() -> list[tuple[str, int | None]]:
    """Labels and indexes for the dashboard picker; None means automatic."""
    choices = [(AUTO_CAMERA_LABEL, None)]
    names = camera_names()
    if names:
        for index, name in enumerate(names):
            info = CameraInfo(index, name)
            choices.append((info.label, index))
    else:
        for index in range(MAX_CAMERA_INDEX):
            choices.append((f"Camera {index}", index))
    return choices


def _camera_info(index: int) -> CameraInfo:
    names = camera_names()
    if 0 <= index < len(names):
        return CameraInfo(index, names[index])
    return CameraInfo(index, f"Camera {index}", named=False)


def _auto_candidates() -> list[CameraInfo]:
    names = camera_names()
    if names:
        cameras = [CameraInfo(index, name)
                   for index, name in enumerate(names)]
        cameras.sort(key=lambda camera: (not camera.is_usb, camera.index))
        return cameras
    # Without names we cannot tell the cameras apart, so try the higher
    # indexes first (the external camera usually appears after the built-in
    # one) and use index 0 last.
    return ([CameraInfo(index, f"Camera {index}", named=False)
             for index in range(1, MAX_CAMERA_INDEX)]
            + [CameraInfo(0, "Camera 0", named=False)])


def _open_index(index: int, width: int, height: int):
    backends = [cv2.CAP_DSHOW]
    msmf = getattr(cv2, "CAP_MSMF", None)
    if msmf is not None:
        backends.append(msmf)
    for backend in backends:
        camera = cv2.VideoCapture(index, backend)
        if not camera.isOpened():
            camera.release()
            continue
        try:
            camera.set(cv2.CAP_PROP_FOURCC,
                       cv2.VideoWriter_fourcc(*"MJPG"))
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            for _ in range(3):
                ok, frame = camera.read()
                if ok and frame is not None:
                    return camera
        except Exception:
            pass
        camera.release()
    return None


def open_camera(preferred_index=None, width=640, height=480):
    """Open the preferred camera, or the USB camera first.

    Returns (cv2.VideoCapture, CameraInfo). Raises CameraError when no
    camera can be opened.
    """
    if preferred_index is None:
        candidates = _auto_candidates()
    else:
        candidates = [_camera_info(int(preferred_index))]

    for info in candidates:
        camera = _open_index(info.index, width, height)
        if camera is not None:
            return camera, info

    if preferred_index is not None:
        raise CameraError(
            f"Camera {preferred_index} could not be opened. Close any other "
            "app that is using it, or choose Auto / another camera in the "
            "dashboard."
        )
    raise CameraError(
        "No working camera found. Connect the USB camera and close any "
        "other app that is using it."
    )
