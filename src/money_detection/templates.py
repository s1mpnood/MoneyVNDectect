"""Template utilities for Vietnamese banknote detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


@dataclass
class BanknoteTemplate:
    """Represents a single banknote template used during detection."""

    label: str
    image: np.ndarray
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    color_histogram: np.ndarray

    @classmethod
    def from_file(
        cls,
        path: Path,
        orb: cv2.ORB,
        histogram_bins: Tuple[int, int, int] = (8, 8, 8),
    ) -> "BanknoteTemplate":
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Unable to load template image: {path}")
        keypoints, descriptors = orb.detectAndCompute(image, None)
        histogram = _compute_normalized_histogram(image, histogram_bins)
        return cls(
            label=path.stem,
            image=image,
            keypoints=keypoints,
            descriptors=descriptors,
            color_histogram=histogram,
        )


class TemplateLibrary:
    """Container for a collection of banknote templates."""

    def __init__(self, templates: Iterable[BanknoteTemplate]):
        self._templates = list(templates)
        if not self._templates:
            raise ValueError("TemplateLibrary requires at least one template")

    @classmethod
    def from_directory(
        cls,
        directory: Path,
        orb: cv2.ORB,
        glob_pattern: str = "*.jpg",
    ) -> "TemplateLibrary":
        if not directory.exists():
            raise FileNotFoundError(f"Template directory not found: {directory}")
        template_paths = sorted(directory.glob(glob_pattern))
        if not template_paths:
            raise FileNotFoundError(
                f"No template images matching {glob_pattern} in {directory}"
            )
        templates = [BanknoteTemplate.from_file(path, orb) for path in template_paths]
        return cls(templates)

    def __iter__(self):
        return iter(self._templates)

    def __len__(self) -> int:
        return len(self._templates)

    def as_dict(self) -> Dict[str, BanknoteTemplate]:
        return {template.label: template for template in self._templates}


def _compute_normalized_histogram(
    image: np.ndarray, bins: Tuple[int, int, int]
) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    histogram = cv2.normalize(histogram, histogram).flatten()
    return histogram
