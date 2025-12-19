import os
import glob
import cv2
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DENOMS = ["10k", "20k", "50k", "100k", "200k", "500k"]
TEMPLATES_DIR = "templates"
IMAGE_SIZE = (200, 100)

MONEY_COLORS = {
    "10k": [(90, 50, 50), (130, 255, 255)],
    "20k": [(90, 50, 50), (130, 255, 255)],
    "50k": [(35, 50, 50), (85, 255, 255)],
    "100k": [(35, 50, 50), (85, 255, 255)],
    "200k": [(0, 50, 50), (20, 255, 255)],
    "500k": [(100, 50, 50), (140, 255, 255)],
}


def _log_debug(msg):
    if os.environ.get("MONEY_DEBUG"):
        print("[DEBUG]", msg)


def _get_image_paths(folder):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    return paths


def preprocess_image(img):
    if img is None or img.size == 0:
        return img
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return enhanced
    except Exception as e:
        _log_debug(f"Preprocess error: {e}")
        return img


def detect_money(img):
    """Detect banknote region using edge detection and aspect ratio filtering."""
    if img is None or img.size == 0:
        return img
    
    h_img, w_img = img.shape[:2]
    
    detect_width = 640
    scale = 1.0
    if w_img > detect_width:
        scale = detect_width / w_img
        work_img = cv2.resize(img, (detect_width, int(h_img * scale)))
    else:
        work_img = img
        
    h_work, w_work = work_img.shape[:2]
    min_area = h_work * w_work * 0.1
    
    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    best_contour = None
    best_area = 0
    
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges_canny = cv2.Canny(blurred, 50, 150)
    thresh_adapt = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 5)
    
    candidates = [thresh_otsu, edges_canny, thresh_adapt]
    for binary_map in candidates:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0
            
            if 1.2 < aspect_ratio < 3.5 and area > best_area:
                best_area = area
                best_contour = (x, y, w, h)
    
    if best_contour:
        x, y, w, h = best_contour
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)
        
        pad = 5
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(w_img - x, w + 2 * pad)
        h = min(h_img - y, h + 2 * pad)
        return img[y:y+h, x:x+w]
    
    cy, cx = h_img // 2, w_img // 2
    ch, cw = int(h_img * 0.8), int(w_img * 0.8)
    y, x = cy - ch//2, cx - cw//2
    return img[y:y+ch, x:x+cw]


def _safe_resize(img, size):
    if img is None or img.size == 0:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    try:
        return cv2.resize(img, size)
    except Exception as e:
        _log_debug(f"Resize fail: {e} size={size}")
        return img


def extract_color_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    hist_h = cv2.calcHist([hsv], [0], None, [36], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)
    
    hist_b = cv2.calcHist([img], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [32], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [32], [0, 256])
    
    cv2.normalize(hist_b, hist_b)
    cv2.normalize(hist_g, hist_g)
    cv2.normalize(hist_r, hist_r)
    
    moments = []
    for ch in cv2.split(hsv):
        moments.extend([np.mean(ch) / 255.0, np.std(ch) / 255.0])
    
    return np.concatenate([
        hist_h.flatten(), hist_s.flatten(), hist_v.flatten(),
        hist_b.flatten(), hist_g.flatten(), hist_r.flatten(),
        np.array(moments)
    ])


def extract_texture_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    angle = cv2.phase(gx, gy, angleInDegrees=True)
    
    if mag.size == 0:
        return np.zeros(64, dtype=np.float32)
    
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hist_mag = cv2.calcHist([mag_norm], [0], None, [32], [0, 256])
    cv2.normalize(hist_mag, hist_mag)
    
    angle_norm = (angle / 360.0 * 255).astype(np.uint8)
    hist_angle = cv2.calcHist([angle_norm], [0], None, [18], [0, 256])
    cv2.normalize(hist_angle, hist_angle)
    
    lbp_features = []
    for ksize in [3, 5]:
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
        lap_norm = cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hist_lap = cv2.calcHist([lap_norm], [0], None, [16], [0, 256])
        cv2.normalize(hist_lap, hist_lap)
        lbp_features.append(hist_lap.flatten())
    
    return np.concatenate([hist_mag.flatten(), hist_angle.flatten()] + lbp_features)


def extract_cnn_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    kernels = [
        np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
        np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
        np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
        np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.0,
    ]
    
    features = []
    for k in kernels:
        filtered = cv2.filter2D(gray, -1, k.astype(np.float32))
        pooled1 = cv2.resize(np.abs(filtered), (10, 5))
        pooled2 = cv2.resize(np.abs(filtered), (5, 3))
        features.append(pooled1.flatten())
        features.append(pooled2.flatten())
    
    return np.concatenate(features)


def extract_shape_features(img):
    """Extract shape-based features specific to banknotes."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corner_count = np.sum(corners > 0.01 * corners.max())
    
    edges = cv2.Canny(gray, 50, 150)
    h, w = edges.shape
    regions = [
        edges[:h//2, :],
        edges[h//2:, :],
        edges[:, :w//2],
        edges[:, w//2:],
        edges[h//4:3*h//4, w//4:3*w//4]
    ]
    edge_densities = [np.mean(r) / 255.0 for r in regions]
    
    return np.array([corner_count / 1000.0] + edge_densities)


def extract_all_features(img):
    if img is None or img.size == 0:
        return np.array([])
    
    img_resized = _safe_resize(img, IMAGE_SIZE)
    
    color = extract_color_features(img_resized)
    texture = extract_texture_features(img_resized)
    cnn = extract_cnn_features(img_resized)
    shape = extract_shape_features(img_resized)
    
    features = np.concatenate([color, texture, cnn, shape])
    norm = np.linalg.norm(features)
    return features / norm if norm > 0 else features


def augment_image(img):
    if img is None or img.size == 0:
        return [img]
    
    augmented = [img]
    h, w = img.shape[:2]
    
    augmented.append(cv2.flip(img, 1))
    
    for angle in [-5, 5]:
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        augmented.append(rotated)
    
    for beta in [-20, 20]:
        bright = cv2.convertScaleAbs(img, alpha=1.0, beta=beta)
        augmented.append(bright)
    
    return augmented


def load_data():
    X, y = [], []
    
    for label in DENOMS:
        folder = os.path.join(TEMPLATES_DIR, label)
        if not os.path.exists(folder):
            print(f"WARNING: Folder not found for {label}")
            continue
        
        paths = _get_image_paths(folder)
        
        if len(paths) == 0:
            print(f"WARNING: No images found for {label}")
            continue
        
        print(f"Loading {label}: {len(paths)} images")
        
        for i, path in enumerate(paths):
            try:
                img = cv2.imread(path)
                if img is None:
                    _log_debug(f"Cannot read {path}")
                    continue
                
                cropped = detect_money(img)
                if cropped is None or cropped.size == 0:
                    cropped = img
                
                cropped = _safe_resize(cropped, IMAGE_SIZE)
                
                preprocessed = preprocess_image(cropped)
                
                for aug_img in augment_image(preprocessed):
                    feat = extract_all_features(aug_img)
                    if feat.size == 0:
                        continue
                    X.append(feat)
                    y.append(label)
                    
            except Exception as e:
                _log_debug(f"Error processing {path}: {e}")
                continue
            
            if (i + 1) % 5 == 0 or (i + 1) == len(paths):
                percent = (i + 1) / len(paths) * 100
                print(f"  Processing {label}: {i + 1}/{len(paths)} ({percent:.1f}%)", end='\r')
        print()
    
    return np.array(X), np.array(y)


def build_orb_templates():
    orb = cv2.ORB_create(nfeatures=1000)
    templates = {}
    
    for label in DENOMS:
        folder = os.path.join(TEMPLATES_DIR, label)
        if not os.path.exists(folder):
            continue
        
        paths = _get_image_paths(folder)
        
        all_des = []
        for path in paths[:10]:
            try:
                img = cv2.imread(path)
                if img is None:
                    continue
                
                cropped = detect_money(img)
                img = preprocess_image(cropped)
                img = _safe_resize(img, IMAGE_SIZE)
                
                kp, des = orb.detectAndCompute(img, None)
                
                if des is not None and len(des) > 0:
                    all_des.append(des)
            except Exception as e:
                _log_debug(f"ORB error {path}: {e}")
                continue
        
        if all_des:
            templates[label] = np.vstack(all_des)
            print(f"ORB {label}: {templates[label].shape[0]} descriptors")
        else:
            print(f"WARNING: No ORB descriptors for {label}")
    
    return templates


def train():
    try:
        print("=" * 50)
        print("Vietnamese Money Recognition - Training")
        print("=" * 50)
        print(f"\nDenominations: {DENOMS}")
        print(f"Templates folder: {TEMPLATES_DIR}")
        print(f"Image size: {IMAGE_SIZE}\n")
        
        print("Loading data...")
        X, y = load_data()
        
        if len(X) == 0:
            print("\nNo data found!")
            print("Please add images to templates/<denomination>/ folders")
            print("Example structure:")
            for d in DENOMS:
                print(f"  templates/{d}/image1.jpg")
            return
        
        print(f"\nTotal samples: {len(X)}")
        print(f"Feature vector size: {X.shape[1]}")
        
        classes_found = []
        for label in DENOMS:
            count = np.sum(y == label)
            if count > 0:
                classes_found.append(label)
            status = ""
            if count == 0:
                status = " (MISSING)"
            elif count < 20:
                status = " (LOW - need more images)"
            print(f"  {label}: {count}{status}")
        
        if len(classes_found) < 2:
            print("\nERROR: Need at least 2 classes to train")
            return
        
        min_count = min(np.sum(y == label) for label in classes_found)
        use_stratify = min_count >= 2
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if use_stratify else None
        )
        
        print(f"\nTraining set: {len(X_train)}")
        print(f"Test set: {len(X_test)}")
        
        print("\nTraining CNN-Feature based Classifier (SVM)...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
        clf.fit(X_train_scaled, y_train)
        
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {acc*100:.2f}%\n")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        print("Building ORB templates...")
        templates = build_orb_templates()
        
        model = {
            "clf": clf,
            "scaler": scaler,
            "templates": templates,
            "image_size": IMAGE_SIZE,
            "feature_length": int(X.shape[1]),
            "classes": list(clf.classes_),
            "accuracy": float(acc)
        }
        
        with open("money_model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        print(f"\n{'=' * 50}")
        print(f"Model saved to money_model.pkl")
        print(f"Classes: {model['classes']}")
        print(f"Accuracy: {acc*100:.2f}%")
        print(f"{'=' * 50}")
        
    except Exception as e:
        import traceback
        print(f"\nTraining error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    train()
