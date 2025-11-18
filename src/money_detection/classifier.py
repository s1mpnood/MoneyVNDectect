"""Hybrid classifier combining color-texture features with a CNN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
import torch
from torch import nn


@dataclass
class ColorTextureFeatures:
    """Feature vector computed from color histogram and LBP texture."""

    histogram: np.ndarray
    lbp: np.ndarray

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.histogram, self.lbp])


class _SimpleCNN(nn.Module):
    """Lightweight CNN suitable for fine-tuning on banknote crops."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class HybridClassifier:
    """Combines handcrafted features with a CNN for banknote classification."""

    def __init__(
        self,
        labels: Iterable[str],
        histogram_bins: Tuple[int, int, int] = (8, 8, 8),
        lbp_radius: int = 2,
        lbp_points: int = 24,
        device: str = "cpu",
    ) -> None:
        self.labels = sorted(set(labels))
        self.label_to_index: Dict[str, int] = {label: idx for idx, label in enumerate(self.labels)}
        self.index_to_label: Dict[int, str] = {idx: label for label, idx in self.label_to_index.items()}
        self.histogram_bins = histogram_bins
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points
        self.device = torch.device(device)
        self.model = _SimpleCNN(num_classes=len(self.labels)).to(self.device)

    def load_weights(self, path: Path) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def extract_features(self, image: np.ndarray) -> ColorTextureFeatures:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            self.histogram_bins,
            [0, 180, 0, 256, 0, 256],
        )
        histogram = cv2.normalize(histogram, histogram).flatten()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = _local_binary_pattern(gray, self.lbp_points, self.lbp_radius)
        hist_lbp, _ = np.histogram(lbp, bins=np.arange(0, self.lbp_points + 3), density=True)

        return ColorTextureFeatures(histogram=histogram, lbp=hist_lbp.astype(np.float32))

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        features = self.extract_features(image)
        handcrafted_vector = features.as_vector()
        handcrafted_scores = self._handcrafted_scores(handcrafted_vector)

        tensor = self._prepare_tensor(image)
        with torch.no_grad():
            logits = self.model(tensor)
            cnn_scores = torch.softmax(logits, dim=1).cpu().numpy()[0]

        combined = 0.4 * handcrafted_scores + 0.6 * cnn_scores
        best_index = int(np.argmax(combined))
        return self.index_to_label[best_index], float(combined[best_index])

    def _handcrafted_scores(self, vector: np.ndarray) -> np.ndarray:
        # In practice you would train a shallow classifier; here we normalize as proxy.
        normalized = vector / (np.linalg.norm(vector) + 1e-8)
        prototypes = np.eye(len(self.labels))
        scores = normalized[: len(self.labels)] if normalized.shape[0] >= len(self.labels) else np.pad(
            normalized, (0, len(self.labels) - normalized.shape[0]), constant_values=0
        )
        return scores[: len(self.labels)]

    def _prepare_tensor(self, image: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(image, (128, 64))
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        return tensor


def _local_binary_pattern(image: np.ndarray, points: int, radius: int) -> np.ndarray:
    lbp = np.zeros_like(image, dtype=np.uint8)
    for idx in range(points):
        theta = 2.0 * np.pi * idx / points
        x = radius * np.cos(theta)
        y = -radius * np.sin(theta)
        shifted = _bilinear_interpolate(image, x, y)
        lbp |= ((shifted >= image).astype(np.uint8) << idx)
    return lbp


def _bilinear_interpolate(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    height, width = image.shape
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    coords_x = np.clip(grid_x + dx, 0, width - 1)
    coords_y = np.clip(grid_y + dy, 0, height - 1)

    x0 = np.floor(coords_x).astype(int)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y0 = np.floor(coords_y).astype(int)
    y1 = np.clip(y0 + 1, 0, height - 1)

    wa = (x1 - coords_x) * (y1 - coords_y)
    wb = (coords_x - x0) * (y1 - coords_y)
    wc = (x1 - coords_x) * (coords_y - y0)
    wd = (coords_x - x0) * (coords_y - y0)

    interpolated = (
        wa * image[y0, x0]
        + wb * image[y0, x1]
        + wc * image[y1, x0]
        + wd * image[y1, x1]
    )
    return interpolated.astype(image.dtype)
