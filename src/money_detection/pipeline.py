"""End-to-end pipeline for Vietnamese banknote detection and classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .classifier import HybridClassifier
from .orb_detector import ORBDetector, DetectionResult
from .templates import TemplateLibrary


@dataclass
class BanknotePrediction:
    label: str
    confidence: float
    polygon: np.ndarray


class DetectionPipeline:
    """Combine ORB detection with hybrid classification on detected crops."""

    def __init__(
        self,
        template_dir: Path,
        labels: List[str],
        classifier_weights: Optional[Path] = None,
        device: str = "cpu",
    ) -> None:
        orb = cv2.ORB_create()
        self.templates = TemplateLibrary.from_directory(template_dir, orb)
        self.detector = ORBDetector(list(self.templates))
        self.classifier = HybridClassifier(labels=labels, device=device)
        if classifier_weights is not None:
            self.classifier.load_weights(classifier_weights)

    def __call__(self, image: np.ndarray) -> List[BanknotePrediction]:
        detections = self.detector.detect(image)
        predictions: List[BanknotePrediction] = []
        for detection in detections:
            crop = self._crop_polygon(image, detection.quad)
            label, confidence = self.classifier.predict(crop)
            predictions.append(
                BanknotePrediction(label=label, confidence=confidence, polygon=detection.quad)
            )
        return predictions

    def _crop_polygon(self, image: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        rect = cv2.boundingRect(polygon.astype(np.float32))
        x, y, w, h = rect
        cropped = image[y : y + h, x : x + w]
        return cropped


def visualize_predictions(
    image: np.ndarray,
    predictions: List[BanknotePrediction],
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    annotated = image.copy()
    for prediction in predictions:
        pts = prediction.polygon.astype(int)
        cv2.polylines(annotated, [pts], True, color, 2)
        x, y = pts[0, 0]
        text = f"{prediction.label}: {prediction.confidence:.2f}"
        cv2.putText(annotated, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated
