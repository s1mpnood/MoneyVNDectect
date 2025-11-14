"""Simple Flask application for Vietnamese banknote classification."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from flask import Flask, jsonify, redirect, render_template_string, request, url_for

from money_detection.pipeline import DetectionPipeline, visualize_predictions, BanknotePrediction


def _load_image_from_bytes(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Không thể đọc ảnh tải lên. Vui lòng kiểm tra định dạng (jpg/png).")
    return image


def _infer_labels(template_dir: Path) -> List[str]:
    labels = [p.stem for p in template_dir.glob("*") if p.is_file()]
    if not labels:
        raise ValueError(
            "Không tìm thấy ảnh mẫu nào trong thư mục template. "
            "Hãy kiểm tra biến môi trường BANKNOTE_TEMPLATE_DIR."
        )
    return labels


def _load_pipeline() -> DetectionPipeline:
    template_dir = Path(os.environ.get("BANKNOTE_TEMPLATE_DIR", "templates"))
    if not template_dir.exists():
        raise RuntimeError(
            "Thư mục template không tồn tại. Đặt biến môi trường BANKNOTE_TEMPLATE_DIR "
            "hoặc tạo thư mục 'templates/'."
        )

    labels_env = os.environ.get("BANKNOTE_LABELS")
    labels = [label.strip() for label in labels_env.split(",") if label.strip()] if labels_env else _infer_labels(template_dir)

    weights_env = os.environ.get("BANKNOTE_WEIGHTS")
    weights: Optional[Path] = Path(weights_env) if weights_env else None

    return DetectionPipeline(template_dir=template_dir, labels=labels, classifier_weights=weights)


PIPELINE = _load_pipeline()
APP = Flask(__name__)


def _prediction_to_dict(prediction: BanknotePrediction) -> dict:
    return {
        "label": prediction.label,
        "confidence": prediction.confidence,
        "polygon": prediction.polygon.astype(float).tolist(),
    }


@APP.route("/", methods=["GET"])
def index() -> str:
    return render_template_string(
        """
        <!doctype html>
        <title>Nhận dạng tiền Việt Nam</title>
        <h1>Tải ảnh tờ tiền để phân loại</h1>
        <form method=post enctype=multipart/form-data action="{{ url_for('predict') }}">
          <input type=file name=image accept="image/*" required>
          <input type=submit value="Phân loại">
        </form>
        """
    )


@APP.route("/predict", methods=["POST"])
def predict() -> str:
    file = request.files.get("image")
    if not file or file.filename == "":
        return redirect(url_for("index"))

    try:
        image = _load_image_from_bytes(file.read())
    except ValueError as exc:
        return render_template_string("<p>{{ message }}</p>", message=str(exc)), 400

    predictions = PIPELINE(image)
    annotated = visualize_predictions(image, predictions)
    _, buffer = cv2.imencode(".jpg", annotated)
    b64 = base64.b64encode(buffer).decode("utf-8")

    return render_template_string(
        """
        <!doctype html>
        <title>Kết quả phân loại</title>
        <h1>Kết quả phân loại</h1>
        <a href="{{ url_for('index') }}">&larr; Quay lại</a>
        <ul>
          {% for pred in predictions %}
            <li>{{ pred.label }} ({{ '%.2f'|format(pred.confidence) }})</li>
          {% else %}
            <li>Không phát hiện được tờ tiền nào.</li>
          {% endfor %}
        </ul>
        <img src="data:image/jpeg;base64,{{ image_base64 }}" alt="Kết quả" style="max-width: 100%; height: auto;" />
        """,
        predictions=predictions,
        image_base64=b64,
    )


@APP.route("/api/predict", methods=["POST"])
def predict_api():
    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"error": "Thiếu ảnh tải lên."}), 400

    try:
        image = _load_image_from_bytes(file.read())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    predictions = [_prediction_to_dict(pred) for pred in PIPELINE(image)]
    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    APP.run(host="0.0.0.0", port=port)
