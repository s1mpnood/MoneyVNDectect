# recognize_image_simple.py
import cv2
import numpy as np
import pickle
import os

from tkinter import Tk, filedialog, Label, Button, Frame, BOTH, TOP, BOTTOM
from PIL import Image, ImageTk

MODEL_PATH = "money_model.pkl"


# =============== FEATURE: COLOR ===============
def extract_color_hist(img, image_size):
    img = cv2.resize(img, image_size)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None,
                        [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# =============== FEATURE: TEXTURE ============
def extract_texture_feature(img, image_size):
    img = cv2.resize(img, image_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_u8 = np.uint8(np.clip(mag, 0, 255))
    hist = cv2.calcHist([mag_u8], [0], None, [16], [0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# =============== FEATURE: MINI CNN ===========
def mini_cnn_feature(img, image_size):
    img = cv2.resize(img, image_size)
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
        f[f < 0] = 0         # ReLU
        pooled = cv2.resize(f, (16, 8))   # "pooling" đơn giản
        feat_maps.append(pooled.flatten())

    return np.concatenate(feat_maps)


def extract_features(img, image_size):
    c = extract_color_hist(img, image_size)
    t = extract_texture_feature(img, image_size)
    cnn = mini_cnn_feature(img, image_size)
    return np.concatenate([c, t, cnn])


# =============== ORB VOTE ====================
def orb_vote(img, templates, image_size, orb, bf):
    img = cv2.resize(img, image_size)
    kp, des = orb.detectAndCompute(img, None)
    if des is None:
        return None, 0

    best_label = None
    best_score = 0

    for label, temp_des in templates.items():
        if temp_des is None:
            continue

        matches = bf.match(des, temp_des)
        matches = sorted(matches, key=lambda x: x.distance)[:30]
        good = [m for m in matches if m.distance < 60]
        score = len(good)

        if score > best_score:
            best_score = score
            best_label = label

    return best_label, best_score


# =============== CLASSIFY 1 ẢNH ==============
def classify_image(path, clf, templates, image_size, orb, bf):
    img = cv2.imread(path)
    if img is None:
        print("Không đọc được ảnh:", path)
        return None, "Không đọc được ảnh!"

    # Ảnh dùng để hiển thị (resize cho vừa màn hình)
    show = img.copy()
    h, w = show.shape[:2]
    max_w = 800
    if w > max_w:
        scale = max_w / w
        show = cv2.resize(show, (int(w * scale), int(h * scale)))

    # ---- KNN với feature màu + texture + mini-CNN ----
    feat = extract_features(img, image_size)
    feat = feat.reshape(1, -1)
    proba = clf.predict_proba(feat)[0]
    idx = np.argmax(proba)
    label_knn = clf.classes_[idx]
    p_knn = proba[idx]

    # ---- ORB template matching ----
    label_orb, score_orb = orb_vote(img, templates, image_size, orb, bf)

    # ---- Fusion: ưu tiên ORB nếu match tốt, nếu không thì KNN ----
    final_label = label_knn
    if label_orb is not None and score_orb >= 25:
        final_label = label_orb
    elif p_knn < 0.60:
        final_label = "Unknown"

    text = f"{final_label} | KNN={label_knn}({p_knn:.2f}) ORB={label_orb}({score_orb})"
    print("Ảnh:", os.path.basename(path))
    print("  ->", text)

    # Vẽ text lên ảnh để hiển thị đẹp hơn
    cv2.putText(show, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0), 2, cv2.LINE_AA)

    return show, text


# =============== TKINTER GUI =================
def main():
    # Load model
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    clf = data["clf"]
    templates = data["templates"]
    image_size = data["image_size"]

    orb = cv2.ORB_create(500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # ----- Tạo cửa sổ Tkinter -----
    root = Tk()
    root.title("Nhận dạng mệnh giá tiền Việt Nam")

    # Frame hiển thị ảnh
    image_frame = Frame(root)
    image_frame.pack(side=TOP, fill=BOTH, expand=True)

    image_label = Label(image_frame)
    image_label.pack(side=TOP, padx=10, pady=10)

    # Label hiển thị kết quả
    result_label = Label(root, text="Chưa có ảnh", font=("Arial", 14))
    result_label.pack(side=TOP, pady=5)

    # Hàm xử lý khi bấm nút "Chọn ảnh"
    def on_choose_image():
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh tờ tiền",
            filetypes=(
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp"),
                ("All files", "*.*"),
            )
        )
        if not file_path:
            return

        show_bgr, text = classify_image(file_path, clf, templates, image_size, orb, bf)
        if show_bgr is None:
            result_label.config(text=text)
            return

        # Cập nhật result label
        result_label.config(text=text)

        # Chuyển BGR (cv2) sang RGB (PIL)
        show_rgb = cv2.cvtColor(show_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(show_rgb)
        # Resize nhỏ lại nếu cần (cho chắc chắn không quá to)
        max_w = 800
        if img_pil.width > max_w:
            scale = max_w / img_pil.width
            new_size = (int(img_pil.width * scale), int(img_pil.height * scale))
            img_pil = img_pil.resize(new_size, Image.ANTIALIAS)

        img_tk = ImageTk.PhotoImage(img_pil)
        image_label.config(image=img_tk)
        image_label.image = img_tk  # giữ reference, tránh bị GC

    # Nút chọn ảnh
    btn = Button(root, text="Chọn ảnh tờ tiền", command=on_choose_image,
                 font=("Arial", 12))
    btn.pack(side=BOTTOM, pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
