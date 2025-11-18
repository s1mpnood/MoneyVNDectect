"""Command line interface for detecting Vietnamese banknotes in images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2

from money_detection.pipeline import DetectionPipeline, visualize_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vietnamese banknote detection")
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument(
        "--templates",
        type=Path,
        required=True,
        help="Directory containing template images (e.g., 10k.jpg)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="List of supported banknote labels (e.g., 10k 20k 50k)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional path to trained CNN weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for inference (cpu or cuda)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save annotated output image",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Unable to load input image: {args.image}")

    pipeline = DetectionPipeline(
        template_dir=args.templates,
        labels=args.labels,
        classifier_weights=args.weights,
        device=args.device,
    )

    predictions = pipeline(image)
    annotated = visualize_predictions(image, predictions)

    if args.output is not None:
        cv2.imwrite(str(args.output), annotated)
    else:
        cv2.imshow("Banknote Detection", annotated)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
