import cv2
import numpy as np
import pickle
import os
from tkinter import Tk, filedialog, Label, Button, Frame, TOP, BOTTOM
from PIL import Image, ImageTk

MODEL_PATH = "money_model.pkl"


def detect_money(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img
    
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    return img[y:y+h, x:x+w]


def preprocess_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def extract_color_features(img, size):
    img = cv2.resize(img, size)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    hist_h = cv2.calcHist([hsv], [0], None, [30], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)
    
    return np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])


def extract_texture_features(img, size):
    img = cv2.resize(img, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    
    hist = cv2.calcHist([mag.astype(np.uint8)], [0], None, [32], [0, 256])
    cv2.normalize(hist, hist)
    
    return hist.flatten()


def extract_cnn_features(img, size):
    img = cv2.resize(img, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    
    kernels = [
        np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
        np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
        np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    ]
    
    features = []
    for k in kernels:
        filtered = cv2.filter2D(gray, -1, k.astype(np.float32))
        pooled = cv2.resize(np.abs(filtered), (10, 5))
        features.append(pooled.flatten())
    
    return np.concatenate(features)


def extract_all_features(img, size):
    color = extract_color_features(img, size)
    texture = extract_texture_features(img, size)
    cnn = extract_cnn_features(img, size)
    
    features = np.concatenate([color, texture, cnn])
    norm = np.linalg.norm(features)
    return features / norm if norm > 0 else features


def orb_match(img, templates, size):
    img = cv2.resize(img, size)
    orb = cv2.ORB_create(nfeatures=500)
    kp, des = orb.detectAndCompute(img, None)
    
    if des is None:
        return None, 0
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    best_label = None
    best_score = 0
    
    for label, temp_des in templates.items():
        matches = bf.match(des, temp_des)
        score = len([m for m in matches if m.distance < 50])
        
        if score > best_score:
            best_score = score
            best_label = label
    
    return best_label, best_score


def classify_image(path, clf, templates, image_size):
    img = cv2.imread(path)
    if img is None:
        return None, "Cannot read image"
    
    detected = detect_money(img)
    preprocessed = preprocess_image(detected)
    
    feat = extract_all_features(preprocessed, image_size)
    proba = clf.predict_proba([feat])[0]
    
    sorted_idx = np.argsort(proba)[::-1]
    label_knn = clf.classes_[sorted_idx[0]]
    p_knn = proba[sorted_idx[0]]
    p_second = proba[sorted_idx[1]]
    
    label_orb, score_orb = orb_match(preprocessed, templates, image_size)
    
    final_label = "Unknown"
    confidence = 0
    
    if p_knn < 0.4:
        final_label = "Unknown"
        confidence = 0
        reason = "KNN confidence too low"
    
    elif (p_knn - p_second) < 0.15 and score_orb < 15:
        final_label = "Unknown"
        confidence = 0
        reason = "KNN uncertain and ORB weak"
    
    elif label_orb is None or score_orb < 10:
        if p_knn >= 0.75 and (p_knn - p_second) >= 0.25:
            final_label = label_knn
            confidence = p_knn * 0.85
            reason = "Trust KNN (ORB failed)"
        else:
            final_label = "Unknown"
            confidence = 0
            reason = "No ORB matches"
    
    elif score_orb >= 40 and label_orb == label_knn:
        final_label = label_orb
        confidence = min((p_knn + min(score_orb/50, 1.0)) / 2, 0.95)
        reason = "ORB and KNN agree strongly"
    
    elif score_orb >= 25 and label_orb == label_knn:
        final_label = label_orb
        confidence = (p_knn * 0.6 + min(score_orb/50, 1.0) * 0.4)
        reason = "ORB and KNN agree"
    
    elif score_orb >= 35:
        if p_knn >= 0.6:
            final_label = "Unknown"
            confidence = 0
            reason = "ORB and KNN conflict"
        else:
            final_label = label_orb
            confidence = min(score_orb/50, 0.8)
            reason = "Trust ORB (KNN weak)"
    
    elif p_knn >= 0.80 and (p_knn - p_second) >= 0.30:
        final_label = label_knn
        confidence = p_knn * 0.90
        reason = "KNN very confident"
    
    elif p_knn >= 0.65:
        if label_orb and score_orb >= 15:
            if label_orb != label_knn:
                final_label = "Unknown"
                confidence = 0
                reason = "ORB-KNN mismatch"
            else:
                final_label = label_knn
                confidence = p_knn * 0.85
                reason = "KNN + weak ORB support"
        else:
            final_label = label_knn
            confidence = p_knn * 0.75
            reason = "Trust KNN only"
    
    else:
        final_label = "Unknown"
        confidence = 0
        reason = "Not enough confidence"
    
    show = cv2.resize(detected, (400, 200))
    
    if final_label == "Unknown":
        text = "Unknown / Not Money"
        color = (0, 0, 255)
    else:
        text = f"{final_label} ({confidence:.0%})"
        color = (0, 255, 0) if confidence > 0.6 else (0, 165, 255)
    
    cv2.putText(show, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(show, f"KNN:{label_knn}({p_knn:.2f}) ORB:{label_orb}({score_orb})", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    cv2.putText(show, reason, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    
    print(f"Result: {text}")
    print(f"KNN: {label_knn} ({p_knn:.2f}), 2nd: {clf.classes_[sorted_idx[1]]} ({p_second:.2f})")
    print(f"ORB: {label_orb} ({score_orb})")
    print(f"Reason: {reason}\n")
    
    return show, text


def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found! Run train.py first")
        return
    
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    
    clf = data["clf"]
    templates = data["templates"]
    image_size = data["image_size"]
    
    root = Tk()
    root.title("Vietnamese Money Recognition")
    root.geometry("600x500")
    
    image_label = Label(root, text="No image", bg="gray")
    image_label.pack(padx=10, pady=10, fill="both", expand=True)
    
    result_label = Label(root, text="Click 'Choose Image' to start", font=("Arial", 12))
    result_label.pack(pady=5)
    
    def choose_image():
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if not path:
            return
        
        result_label.config(text="Processing...")
        root.update()
        
        show, text = classify_image(path, clf, templates, image_size)
        if show is None:
            result_label.config(text=text)
            return
        
        result_label.config(text=text)
        
        img_rgb = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        image_label.config(image=img_tk, text="")
        image_label.image = img_tk
    
    Button(root, text="Choose Image", command=choose_image, 
           font=("Arial", 12), bg="lightblue", padx=20, pady=10).pack(pady=10)
    
    root.mainloop()


if __name__ == "__main__":
    main()
