from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import Detection, Hazard, OCRText, SceneAnalysis


@dataclass
class Perception:
    """One frame's worth of fused perception."""
    scene: SceneAnalysis | None
    detections: list[Detection]
    ocr: list[OCRText]
    hazards: list[Hazard]
    timings: dict[str, float] = field(default_factory=dict)
