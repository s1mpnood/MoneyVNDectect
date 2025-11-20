import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# Auto set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# =====================================================
# FIG → ARRAY
# =====================================================
def fig_to_array(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    arr = buf.reshape(h, w, 4)[:, :, :3]
    return arr


# =====================================================
# FEATURE FUNCTIONS
# =====================================================
def mini_cnn_feature(img):
    img = cv2.resize(img, (256, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobelX = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    sobelY = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    sobel_mag = cv2.magnitude(sobelX, sobelY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)

    hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()

    fig = plt.figure(figsize=(2, 1))
    ax = fig.add_subplot(111)
    ax.imshow(sobel_mag, cmap="hot")
    ax.axis("off")
    heat = fig_to_array(fig)
    plt.close(fig)

    heat_small = cv2.resize(heat, (32, 16)).flatten()
    pooled = cv2.resize(gray, (32, 16)).flatten()

    f = np.concatenate([sobel_mag.flatten(), lap.flatten(), hist,
                        heat_small, pooled]).astype(np.float32)

    return f / (np.linalg.norm(f) + 1e-6)


def cosine_similarity(a, b):
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0
    return dot / (na * nb)


def color_feature(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    hist = np.concatenate([h, s, v]).flatten()
    return cv2.normalize(hist, None).flatten()


def orb_extract(img, orb):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)
    return des


def orb_match_score(des_tpl, des_test, bf):
    if des_tpl is None or des_test is None:
        return 0
    matches = bf.knnMatch(des_tpl, des_test, 2)
    good = 0
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good += 1
    return good


# =====================================================
# LOAD ALL TEMPLATES
# =====================================================
def scan_template_folders():
    samples = {}
    base = "templates"

    for denom in sorted(os.listdir(base)):
        folder = os.path.join(base, denom)
        if not os.path.isdir(folder):
            continue

        imgs = []
        for f in sorted(os.listdir(folder)):
            if f.endswith((".jpg", ".png", ".jpeg")):
                imgs.append(os.path.join(folder, f))

        if imgs:
            samples[denom] = imgs

    return samples


def load_templates(samples):
    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    TPL = {}

    for denom, paths in samples.items():
        TPL[denom] = {"orb": [], "color": [], "cnn": []}

        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue

            TPL[denom]["orb"].append(orb_extract(img, orb))
            TPL[denom]["color"].append(color_feature(img))
            TPL[denom]["cnn"].append(mini_cnn_feature(img))

    return TPL, orb, bf


# =====================================================
# DETECT FUNCTION
# =====================================================
def detect_money(test_path):
    img = cv2.imread(test_path)
    if img is None:
        result_label.config(text="Không đọc được ảnh!")
        return

    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_display = ImageTk.PhotoImage(Image.fromarray(img_display).resize((350, 230)))
    image_label.config(image=img_display)
    image_label.image = img_display

    des_test = orb_extract(img, orb)
    col_test = color_feature(img)
    cnn_test = mini_cnn_feature(img)

    best = None
    best_score = -9999
    best_orb = best_col = best_cnn = 0

    for denom, data in TPL.items():
        orb_s = sum(orb_match_score(d, des_test, bf) for d in data["orb"])
        col_s = sum(cv2.compareHist(h, col_test, cv2.HISTCMP_CORREL) for h in data["color"])
        cnn_s = sum(cosine_similarity(c, cnn_test) for c in data["cnn"])

        total = orb_s + col_s * 3 + cnn_s * 5

        if total > best_score:
            best = denom
            best_score = total
            best_orb, best_col, best_cnn = orb_s, col_s, cnn_s

    result_str = (
        f"Mệnh giá: {best}\n"
        f"ORB:   {best_orb:.2f}\n"
        f"COLOR: {best_col:.2f}\n"
        f"CNN:   {best_cnn:.2f}\n"
        f"TOTAL: {best_score:.2f}"
    )

    result_label.config(text=result_str)


# =====================================================
# TKINTER GUI
# =====================================================
samples = scan_template_folders()
TPL, orb, bf = load_templates(samples)

root = Tk()
root.title("VN Money Detection – Tkinter GUI")
root.geometry("650x500")

btn = Button(root, text="Chọn ảnh để nhận dạng", font=("Arial", 14),
             command=lambda: detect_money(filedialog.askopenfilename()))
btn.pack(pady=10)

image_label = Label(root)
image_label.pack()

result_label = Label(root, text="Vui lòng chọn ảnh", font=("Arial", 14),
                     justify="left")
result_label.pack(pady=10)

root.mainloop()
