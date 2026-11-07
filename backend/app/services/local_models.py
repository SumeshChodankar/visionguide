"""On-box models: YOLOv8n (objects) and EasyOCR (text). Both are loaded once,
lazily, and run in a worker thread so they overlap with the Gemini call.
If a package isn't installed the engine reports itself disabled instead of
crashing, so the API still works on a laptop without torch."""
from __future__ import annotations

import logging
import math
import threading

import numpy as np
from PIL import Image

from ..config import Settings
from ..schemas import Detection
from ..utils.geo import clock_from_image_x, side_from_image_x

log = logging.getLogger("visionguide.models")

# COCO classes that matter for a blind pedestrian, with typical real-world
# heights (m) used for a pinhole-camera distance estimate.
RELEVANT_CLASSES: dict[str, float] = {
    "person": 1.7, "chair": 0.9, "bench": 0.8, "couch": 0.85, "dining table": 0.75, "bed": 0.6,
    "potted plant": 0.7, "suitcase": 0.65, "backpack": 0.5, "handbag": 0.3, "bicycle": 1.0,
    "motorcycle": 1.1, "car": 1.5, "bus": 3.0, "truck": 2.5, "dog": 0.5, "fire hydrant": 0.7,
    "stop sign": 2.2, "traffic light": 3.0, "parking meter": 1.3, "toilet": 0.75, "refrigerator": 1.7,
    "umbrella": 1.0, "skateboard": 0.15, "tv": 0.6, "oven": 0.9, "sink": 0.9,
}


class YoloDetector:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = settings.enable_yolo
        self._model = None
        self._lock = threading.Lock()
        self._focal = 0.5 / math.tan(math.radians(settings.camera_vfov_deg / 2))

    def load(self) -> None:
        if not self.enabled or self._model is not None:
            return
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.settings.yolo_model)
            log.info("YOLO model loaded: %s", self.settings.yolo_model)
        except Exception as e:  # noqa: BLE001
            log.warning("YOLO disabled (%s)", e)
            self.enabled = False

    def detect(self, img: Image.Image) -> list[Detection]:
        self.load()
        if not self.enabled or self._model is None:
            return []
        with self._lock:  # ultralytics predictors are not thread-safe
            res = self._model.predict(img, imgsz=480, conf=self.settings.yolo_conf, verbose=False)[0]
        w, h = img.size
        out: list[Detection] = []
        for box in res.boxes:
            label = res.names[int(box.cls[0])]
            if label not in RELEVANT_CLASSES:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            nb = [x1 / w, y1 / h, x2 / w, y2 / h]
            out.append(Detection(
                label=label, confidence=round(float(box.conf[0]), 3), bbox=[round(v, 4) for v in nb],
                clock=clock_from_image_x((nb[0] + nb[2]) / 2, self.settings.camera_hfov_deg),
                side=side_from_image_x((nb[0] + nb[2]) / 2),
                distance_m=self.estimate_distance(label, nb), source="yolo",
            ))
        return out

    def estimate_distance(self, label: str, nb: list[float]) -> float | None:
        real_h = RELEVANT_CLASSES.get(label)
        h_norm = nb[3] - nb[1]
        if not real_h or h_norm <= 0.01:
            return None
        d = real_h * self._focal / h_norm
        if nb[3] > 0.97 and nb[1] < 0.03:   # object fills the frame: it's right there
            d = min(d, 1.0)
        return round(max(0.3, min(d, 30.0)), 1)


class OcrEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = settings.enable_ocr
        self._reader = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if not self.enabled or self._reader is not None:
            return
        try:
            import easyocr

            langs = [x.strip() for x in self.settings.ocr_languages.split(",") if x.strip()]
            self._reader = easyocr.Reader(langs, gpu=self.settings.ocr_gpu, verbose=False)
            log.info("EasyOCR loaded (%s)", langs)
        except Exception as e:  # noqa: BLE001
            log.warning("EasyOCR disabled (%s)", e)
            self.enabled = False

    def read(self, img: Image.Image) -> list[tuple[list[float], str, float]]:
        """Returns [(normalised bbox x1,y1,x2,y2, text, confidence)]."""
        self.load()
        if not self.enabled or self._reader is None:
            return []
        arr = np.asarray(img.convert("RGB"))
        h, w = arr.shape[:2]
        with self._lock:
            raw = self._reader.readtext(arr, detail=1, paragraph=False)
        out = []
        for pts, text, conf in raw:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if conf < self.settings.ocr_min_conf or not text.strip():
                continue
            out.append(([min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h], text.strip(), float(conf)))
        return out
