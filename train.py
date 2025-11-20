# train_simple.py
import os
import glob
import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

# Các mệnh giá (theo đúng tên folder trong templates/)
DENOMS = ["0k", "10k", "20k", "50k", "100k", "200k", "500k"]
TEMPLATES_DIR = "templates"     # thư mục chứa ảnh từng mệnh giá
IMAGE_SIZE = (300, 100)         # có thể chỉnh lại cho phù hợp


# ==================== COLOR FEATURE ====================
def extract_color_hist(img):
    img = cv2.resize(img, IMAGE_SIZE)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None,
                        [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# ==================== TEXTURE FEATURE ====================
def extract_texture_feature(img):
    img = cv2.resize(img, IMAGE_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = np.uint8(np.clip(mag, 0, 255))
    hist = cv2.calcHist([mag_u8], [0], None, [16], [0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# ==================== MINI-CNN FEATURE ====================
def mini_cnn_feature(img):
    img = cv2.resize(img, IMAGE_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0

    kernels = [
        np.array([[1, 0, -1],
                  [1, 0, -1],
                  [1, 0, -1]], dtype=np.float32),

        np.array([[1, 1, 1],
                  [0, 0, 0],
                  [0, 0, 0]], dtype=np.float32),

        np.array([[0, -1, 0],
                  [-1, 4, -1],
                  [0, -1, 0]], dtype=np.float32),
    ]

    feat_maps = []
    for k in kernels:
        f = cv2.filter2D(gray, -1, k)
        f[f < 0] = 0                # ReLU
        pooled = cv2.resize(f, (16, 8))  # "pooling" đơn giản
        feat_maps.append(pooled.flatten())

    return np.concatenate(feat_maps)


# ==================== GOM TẤT CẢ FEATURE ====================
def extract_features(img):
    c = extract_color_hist(img)
    t = extract_texture_feature(img)
    cnn = mini_cnn_feature(img)
    return np.concatenate([c, t, cnn])


# ==================== LOAD DATASET TỪ templates/ ====================
def load_dataset():
    X = []
    y = []

    for label in DENOMS:
        folder = os.path.join(TEMPLATES_DIR, label)
        if not os.path.isdir(folder):
            print("Bỏ qua (không có folder):", folder)
            continue

        # lấy mọi file ảnh phổ biến
        paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            paths.extend(glob.glob(os.path.join(folder, ext)))

        for path in paths:
            img = cv2.imread(path)
            if img is None:
                continue
            feat = extract_features(img)
            X.append(feat)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print("Tổng số mẫu:", len(y))
    return X, y


# ==================== BUILD ORB TEMPLATES ====================
def build_orb_templates():
    orb = cv2.ORB_create(500)
    templates = {}

    for label in DENOMS:
        folder = os.path.join(TEMPLATES_DIR, label)
        if not os.path.isdir(folder):
            continue

        paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            paths.extend(glob.glob(os.path.join(folder, ext)))

        if not paths:
            continue

        # Lấy ảnh đầu tiên làm template
        img = cv2.imread(paths[0])
        img = cv2.resize(img, IMAGE_SIZE)
        kp, des = orb.detectAndCompute(img, None)
        if des is not None:
            templates[label] = des
            print(f"Tạo ORB template cho {label}, descriptor = {des.shape[0]}")

    return templates


# ==================== TRAIN & SAVE MODEL ====================
def train():
    X, y = load_dataset()

    if len(y) == 0:
        print("Không có dữ liệu để train.")
        return

    clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
    clf.fit(X, y)
    print("Train KNN xong.")

    templates = build_orb_templates()

    model = {
        "clf": clf,
        "denoms": DENOMS,
        "templates": templates,
        "image_size": IMAGE_SIZE,
    }

    with open("money_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Đã lưu model vào money_model.pkl")


if __name__ == "__main__":
    train()
