"""ORB based detection of Vietnamese banknotes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .templates import BanknoteTemplate


@dataclass
class DetectionResult:
    """Describes a detection output from the ORB detector."""

    label: str
    score: float
    quad: np.ndarray


class ORBDetector:
    """Detect banknotes in an image by matching ORB features to templates."""

    def __init__(
        self,
        templates: List[BanknoteTemplate],
        matcher: Optional[cv2.DescriptorMatcher] = None,
        ratio_test: float = 0.75,
        min_matches: int = 10,
    ) -> None:
        self.templates = templates
        self.matcher = matcher or cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.ratio_test = ratio_test
        self.min_matches = min_matches

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)
        if descriptors is None:
            return []

        results: List[DetectionResult] = []
        for template in self.templates:
            matches = self._match_descriptors(descriptors, template.descriptors)
            if len(matches) < self.min_matches:
                continue
            quad, score = self._estimate_pose(template, matches, keypoints)
            if quad is not None:
                results.append(DetectionResult(template.label, score, quad))
        return sorted(results, key=lambda result: result.score, reverse=True)

    def _match_descriptors(
        self,
        descriptors_scene: np.ndarray,
        descriptors_template: np.ndarray,
    ) -> List[cv2.DMatch]:
        raw_matches = self.matcher.knnMatch(descriptors_template, descriptors_scene, k=2)
        matches: List[cv2.DMatch] = []
        for m, n in raw_matches:
            if m.distance < self.ratio_test * n.distance:
                matches.append(m)
        return matches

    def _estimate_pose(
        self,
        template: BanknoteTemplate,
        matches: List[cv2.DMatch],
        keypoints_scene: List[cv2.KeyPoint],
    ) -> tuple[Optional[np.ndarray], float]:
        src_pts = np.float32([template.keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_scene[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        if len(src_pts) < 4:
            return None, 0.0

        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if homography is None or mask is None:
            return None, 0.0
        inliers = mask.ravel().tolist()
        if sum(inliers) < self.min_matches:
            return None, 0.0

        h, w = template.image.shape[:2]
        corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, homography)
        score = float(sum(inliers) / len(matches))
        return projected, score
