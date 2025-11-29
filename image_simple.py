import cv2
import numpy as np
import pickle
import os
from tkinter import Tk, filedialog, Label, Button, Frame, TOP, BOTTOM
from PIL import Image, ImageTk

MODEL_PATH = "money_model.pkl"


def _log_debug(msg):
    if os.environ.get("MONEY_DEBUG"):
        print("[DEBUG]", msg)


def _is_valid_image(img):
    """Check if image is valid for processing."""
    if img is None:
        return False
    if not isinstance(img, np.ndarray):
        return False
    if img.size == 0:
        return False
    if len(img.shape) < 2:
        return False
    return True


def _ensure_bgr(img):
    """Ensure image is in BGR format."""
    if not _is_valid_image(img):
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _safe_resize(img, size):
    if not _is_valid_image(img):
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    try:
        if size[0] <= 0 or size[1] <= 0:
            return img
        return cv2.resize(img, size)
    except Exception as e:
        _log_debug(f"Resize failed for size={size}: {e}")
        return img


def _validate_image_size(sz):
    try:
        if not isinstance(sz, (tuple, list)) or len(sz) != 2:
            raise ValueError("image_size not a 2-sequence")
        w, h = int(sz[0]), int(sz[1])
        if w <= 0 or h <= 0:
            raise ValueError("image_size contains non-positive dimension")
        return (w, h)
    except Exception as e:
        print(f"Invalid image_size ({sz}): {e}. Using fallback (200,100)")
        return (200, 100)


def _safe_slice(img, x, y, w, h):
    """Safely slice image with bounds checking."""
    if not _is_valid_image(img):
        return img
    img_h, img_w = img.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    x2 = max(x + 1, min(x + w, img_w))
    y2 = max(y + 1, min(y + h, img_h))
    result = img[y:y2, x:x2]
    if result.size == 0:
        return img
    return result


def detect_money(img):
    """Detect banknote region using edge detection and aspect ratio filtering."""
    if not _is_valid_image(img):
        return img
    
    img = _ensure_bgr(img)
    if img is None:
        return None
    
    h_img, w_img = img.shape[:2]
    if h_img < 10 or w_img < 10:
        return img
    
    min_area = h_img * w_img * 0.1
    
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        _log_debug(f"Color conversion failed: {e}")
        return img
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    best_contour = None
    best_area = 0
    
    for thresh1, thresh2 in [(30, 100), (50, 150), (20, 80)]:
        edges = cv2.Canny(blurred, thresh1, thresh2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            if h <= 0 or w <= 0:
                continue
            aspect_ratio = float(w) / float(h)
            
            # Vietnamese banknotes have aspect ratio ~2.2:1
            if 1.5 < aspect_ratio < 3.0 and area > best_area:
                best_area = area
                best_contour = (x, y, w, h)
    
    if best_contour:
        x, y, w, h = best_contour
        pad = 5
        return _safe_slice(img, x - pad, y - pad, w + 2 * pad, h + 2 * pad)
    
    return img


def preprocess_image(img):
    if not _is_valid_image(img):
        return img
    
    img = _ensure_bgr(img)
    if img is None:
        return None
    
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    except Exception as e:
        _log_debug(f"Preprocess error: {e}")
        return img


def _safe_histogram(img, channels, bins, ranges):
    """Safely compute histogram with fallback."""
    try:
        hist = cv2.calcHist([img], channels, None, bins, ranges)
        if hist is None or hist.size == 0:
            return np.zeros(bins[0], dtype=np.float32)
        cv2.normalize(hist, hist)
        return hist.flatten()
    except Exception as e:
        _log_debug(f"Histogram error: {e}")
        return np.zeros(bins[0], dtype=np.float32)


def extract_color_features(img, size):
    img = _safe_resize(img, size)
    img = _ensure_bgr(img)
    if img is None:
        return np.zeros(202, dtype=np.float32)  # Expected size
    
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    except Exception:
        return np.zeros(202, dtype=np.float32)
    
    # HSV histograms
    hist_h = _safe_histogram(hsv, [0], [36], [0, 180])
    hist_s = _safe_histogram(hsv, [1], [32], [0, 256])
    hist_v = _safe_histogram(hsv, [2], [32], [0, 256])
    
    # BGR histograms
    hist_b = _safe_histogram(img, [0], [32], [0, 256])
    hist_g = _safe_histogram(img, [1], [32], [0, 256])
    hist_r = _safe_histogram(img, [2], [32], [0, 256])
    
    # Color moments
    moments = []
    try:
        for ch in cv2.split(hsv):
            mean_val = np.mean(ch)
            std_val = np.std(ch)
            # Avoid division issues
            moments.extend([mean_val / 255.0, std_val / 255.0])
    except Exception:
        moments = [0.0] * 6
    
    return np.concatenate([
        hist_h, hist_s, hist_v,
        hist_b, hist_g, hist_r,
        np.array(moments, dtype=np.float32)
    ])


def extract_texture_features(img, size):
    img = _safe_resize(img, size)
    img = _ensure_bgr(img)
    if img is None:
        return np.zeros(82, dtype=np.float32)  # Expected size
    
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        return np.zeros(82, dtype=np.float32)
    
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    angle = cv2.phase(gx, gy, angleInDegrees=True)
    
    if mag.size == 0:
        return np.zeros(82, dtype=np.float32)
    
    # Magnitude histogram
    mag_max = mag.max()
    if mag_max > 0:
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        mag_norm = np.zeros_like(mag, dtype=np.uint8)
    hist_mag = _safe_histogram(mag_norm, [0], [32], [0, 256])
    
    # Gradient orientation histogram
    angle_norm = np.clip(angle / 360.0 * 255, 0, 255).astype(np.uint8)
    hist_angle = _safe_histogram(angle_norm, [0], [18], [0, 256])
    
    # LBP-like texture
    lbp_features = []
    for ksize in [3, 5]:
        try:
            lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
            lap_abs = np.abs(lap)
            if lap_abs.max() > 0:
                lap_norm = cv2.normalize(lap_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                lap_norm = np.zeros_like(lap_abs, dtype=np.uint8)
            hist_lap = _safe_histogram(lap_norm, [0], [16], [0, 256])
            lbp_features.append(hist_lap)
        except Exception:
            lbp_features.append(np.zeros(16, dtype=np.float32))
    
    return np.concatenate([hist_mag, hist_angle] + lbp_features)


def extract_cnn_features(img, size):
    img = _safe_resize(img, size)
    img = _ensure_bgr(img)
    if img is None:
        return np.zeros(150, dtype=np.float32)  # Expected size
    
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    except Exception:
        return np.zeros(150, dtype=np.float32)
    
    kernels = [
        np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32),
        np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32),
        np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32),
        np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32),
        np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32) / 9.0,
    ]
    
    features = []
    for k in kernels:
        try:
            filtered = cv2.filter2D(gray, -1, k)
            filtered_abs = np.abs(filtered)
            pooled1 = cv2.resize(filtered_abs, (10, 5))
            pooled2 = cv2.resize(filtered_abs, (5, 3))
            features.append(pooled1.flatten())
            features.append(pooled2.flatten())
        except Exception:
            features.append(np.zeros(50, dtype=np.float32))
            features.append(np.zeros(15, dtype=np.float32))
    
    return np.concatenate(features)


def extract_shape_features(img, size):
    """Extract shape-based features specific to banknotes."""
    img = _safe_resize(img, size)
    img = _ensure_bgr(img)
    if img is None:
        return np.zeros(6, dtype=np.float32)
    
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        return np.zeros(6, dtype=np.float32)
    
    # Corner detection
    try:
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        corner_max = corners.max()
        if corner_max > 0:
            corner_count = np.sum(corners > 0.01 * corner_max)
        else:
            corner_count = 0
    except Exception:
        corner_count = 0
    
    # Edge density in different regions
    try:
        edges = cv2.Canny(gray, 50, 150)
        h, w = edges.shape
        if h < 4 or w < 4:
            return np.array([corner_count / 1000.0, 0, 0, 0, 0, 0], dtype=np.float32)
        
        regions = [
            edges[:h//2, :],
            edges[h//2:, :],
            edges[:, :w//2],
            edges[:, w//2:],
            edges[h//4:3*h//4, w//4:3*w//4]
        ]
        edge_densities = []
        for r in regions:
            if r.size > 0:
                edge_densities.append(float(np.mean(r)) / 255.0)
            else:
                edge_densities.append(0.0)
    except Exception:
        edge_densities = [0.0] * 5
    
    return np.array([corner_count / 1000.0] + edge_densities, dtype=np.float32)


def extract_all_features(img, size):
    if not _is_valid_image(img):
        return np.array([], dtype=np.float32)
    
    color = extract_color_features(img, size)
    texture = extract_texture_features(img, size)
    cnn = extract_cnn_features(img, size)
    shape = extract_shape_features(img, size)
    
    features = np.concatenate([color, texture, cnn, shape])
    
    # Replace any NaN or Inf values
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm
    
    return features.astype(np.float32)


def orb_match(img, templates, size):
    if not templates:
        return None, 0
    
    img = _safe_resize(img, size)
    img = _ensure_bgr(img)
    if img is None:
        return None, 0
    
    try:
        orb = cv2.ORB_create(nfeatures=1000)
        kp, des = orb.detectAndCompute(img, None)
    except Exception as e:
        _log_debug(f"ORB detection error: {e}")
        return None, 0
    
    if des is None or len(des) == 0:
        return None, 0
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    best_label = None
    best_score = 0
    
    for label, temp_des in templates.items():
        if temp_des is None or len(temp_des) == 0:
            continue
        try:
            matches = bf.match(des, temp_des)
            if not matches:
                continue
            score = sum(1 for m in matches if m.distance < 50)
            if score > best_score:
                best_score = score
                best_label = label
        except Exception as e:
            _log_debug(f"ORB match error for {label}: {e}")
            continue
    
    return best_label, best_score


def _predict_proba_safe(clf, feat):
    try:
        proba = clf.predict_proba([feat])[0]
        classes = clf.classes_
    except AttributeError:
        if hasattr(clf, "decision_function"):
            try:
                df = clf.decision_function([feat])
                df = np.asarray(df)
                if df.ndim == 2:
                    z = df - np.max(df, axis=1, keepdims=True)
                    exp = np.exp(np.clip(z, -500, 500))  # Prevent overflow
                    proba = (exp / exp.sum(axis=1, keepdims=True))[0]
                    classes = getattr(clf, "classes_", np.arange(proba.shape[0]))
                else:
                    p1 = 1 / (1 + np.exp(-np.clip(df[0], -500, 500)))
                    proba = np.array([1 - p1, p1])
                    classes = getattr(clf, "classes_", np.array([0, 1]))
            except Exception:
                proba = np.array([1.0])
                classes = np.array(["Unknown"])
        else:
            proba = np.array([1.0])
            classes = np.array(["Unknown"])
    except Exception:
        proba = np.array([1.0])
        classes = np.array(["Unknown"])
    
    return proba, classes


def classify_image(path, clf, templates, image_size, expected_feature_length=None):
    try:
        img = cv2.imread(path)
        if img is None:
            return None, "Cannot read image"
        
        if not _is_valid_image(img):
            return None, "Invalid image format"
        
        detected = detect_money(img)
        if not _is_valid_image(detected):
            detected = img
        
        preprocessed = preprocess_image(detected)
        if not _is_valid_image(preprocessed):
            preprocessed = detected
        
        feat = extract_all_features(preprocessed, image_size)
        
        if feat.size == 0:
            return None, "Feature extraction failed"
        
        # Validate feature length if provided
        if expected_feature_length is not None and feat.size != expected_feature_length:
            _log_debug(f"Feature length mismatch: got {feat.size}, expected {expected_feature_length}")
            # Pad or truncate to match expected length
            if feat.size < expected_feature_length:
                feat = np.pad(feat, (0, expected_feature_length - feat.size), mode='constant')
            else:
                feat = feat[:expected_feature_length]
        
        proba, classes = _predict_proba_safe(clf, feat)
        if proba.size == 0:
            return None, "Classifier returned empty probabilities"
        
        idx = int(np.argmax(proba))
        label_knn = str(classes[idx])
        p_knn = float(proba[idx])
        
        label_orb, score_orb = orb_match(preprocessed, templates, image_size)
        
        # Decision logic
        if label_orb and score_orb > 20 and label_orb == label_knn:
            final_label = label_orb
            confidence = (p_knn + min(score_orb/50, 1.0)) / 2
        elif p_knn > 0.7:
            final_label = label_knn
            confidence = p_knn
        elif label_orb and score_orb > 30:
            final_label = label_orb
            confidence = min(score_orb/50, 0.9)
        else:
            final_label = label_knn if p_knn > 0.5 else "Unknown"
            confidence = p_knn if p_knn > 0.5 else 0
        
        show = _safe_resize(detected, (400, 200))
        text = f"{final_label} ({confidence:.0%})"
        
        color = (0, 255, 0) if confidence > 0.6 else (0, 165, 255)
        cv2.putText(show, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        orb_str = f"{label_orb}({score_orb})" if label_orb else "None(0)"
        cv2.putText(show, f"KNN:{label_knn}({p_knn:.2f}) ORB:{orb_str}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        print(f"Result: {text}")
        print(f"KNN: {label_knn} ({p_knn:.2f})")
        print(f"ORB: {label_orb} ({score_orb})")
        
        return show, text
    except Exception as e:
        _log_debug(f"Exception in classify_image: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error: {e}"


def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found! Run train.py first")
        return
    
    try:
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    required_keys = {"clf", "templates", "image_size"}
    missing = required_keys - set(data.keys())
    if missing:
        print(f"Model file missing keys: {missing}")
        return
    
    clf = data["clf"]
    templates = data.get("templates") or {}
    image_size = _validate_image_size(data["image_size"])
    expected_feature_length = data.get("feature_length")
    
    print(f"Model loaded. Classes: {data.get('classes', 'N/A')}")
    print(f"Accuracy: {data.get('accuracy', 'N/A')}")
    if expected_feature_length:
        print(f"Feature length: {expected_feature_length}")
    
    root = Tk()
    root.title("Vietnamese Money Recognition")
    root.geometry("600x500")
    
    image_label = Label(root, text="No image", bg="gray")
    image_label.pack(padx=10, pady=10, fill="both", expand=True)
    
    result_label = Label(root, text="Click 'Choose Image' to start", font=("Arial", 12))
    result_label.pack(pady=5)
    
    def choose_image():
        path = filedialog.askopenfilename(filetypes=[
            ("Images", "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG")
        ])
        if not path:
            return
        
        result_label.config(text="Processing...")
        root.update()
        
        try:
            show, text = classify_image(path, clf, templates, image_size, expected_feature_length)
        except Exception as e:
            show, text = None, f"Unhandled error: {e}"
        
        if show is None:
            result_label.config(text=text)
            return
        
        result_label.config(text=text)
        
        try:
            img_rgb = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            image_label.config(image=img_tk, text="")
            image_label.image = img_tk
        except Exception as e:
            result_label.config(text=f"Display error: {e}")
    
    Button(root, text="Choose Image", command=choose_image, 
           font=("Arial", 12), bg="lightblue", padx=20, pady=10).pack(pady=10)
    
    root.mainloop()


if __name__ == "__main__":
    main()
