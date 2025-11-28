import os
import glob
import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DENOMS = ["10k", "20k", "50k", "100k", "200k", "500k"]
TEMPLATES_DIR = "templates"
IMAGE_SIZE = (200, 100)


def preprocess_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return enhanced


def extract_color_features(img):
    img = cv2.resize(img, IMAGE_SIZE)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    hist_h = cv2.calcHist([hsv], [0], None, [30], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    
    cv2.normalize(hist_h, hist_h)
    cv2.normalize(hist_s, hist_s)
    cv2.normalize(hist_v, hist_v)
    
    return np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])


def extract_texture_features(img):
    img = cv2.resize(img, IMAGE_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    
    hist = cv2.calcHist([mag.astype(np.uint8)], [0], None, [32], [0, 256])
    cv2.normalize(hist, hist)
    
    return hist.flatten()


def extract_cnn_features(img):
    img = cv2.resize(img, IMAGE_SIZE)
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


def extract_all_features(img):
    color = extract_color_features(img)
    texture = extract_texture_features(img)
    cnn = extract_cnn_features(img)
    
    features = np.concatenate([color, texture, cnn])
    norm = np.linalg.norm(features)
    return features / norm if norm > 0 else features


def augment_image(img):
    augmented = [img]
    augmented.append(cv2.flip(img, 1))
    
    h, w = img.shape[:2]
    for angle in [-5, 5]:
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
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
            continue
        
        paths = glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png"))
        
        print(f"Loading {label}: {len(paths)} images")
        
        for path in paths:
            img = cv2.imread(path)
            if img is None:
                continue
            
            preprocessed = preprocess_image(img)
            aug_images = augment_image(preprocessed)
            
            for aug_img in aug_images:
                feat = extract_all_features(aug_img)
                X.append(feat)
                y.append(label)
    
    return np.array(X), np.array(y)


def build_orb_templates():
    orb = cv2.ORB_create(nfeatures=500)
    templates = {}
    
    for label in DENOMS:
        folder = os.path.join(TEMPLATES_DIR, label)
        if not os.path.exists(folder):
            continue
        
        paths = glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png"))
        
        all_des = []
        for path in paths[:3]:
            img = cv2.imread(path)
            if img is None:
                continue
            
            img = preprocess_image(img)
            img = cv2.resize(img, IMAGE_SIZE)
            kp, des = orb.detectAndCompute(img, None)
            
            if des is not None:
                all_des.append(des)
        
        if all_des:
            templates[label] = np.vstack(all_des)
            print(f"ORB {label}: {templates[label].shape[0]} descriptors")
    
    return templates


def train():
    print("Loading data...")
    X, y = load_data()
    
    if len(X) == 0:
        print("No data found!")
        return
    
    print(f"\nTotal samples: {len(X)}")
    for label in DENOMS:
        count = np.sum(y == label)
        print(f"  {label}: {count}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining KNN...")
    clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%\n")
    print(classification_report(y_test, y_pred))
    
    print("\nBuilding ORB templates...")
    templates = build_orb_templates()
    
    model = {
        "clf": clf,
        "templates": templates,
        "image_size": IMAGE_SIZE
    }
    
    with open("money_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("\nModel saved to money_model.pkl")


if __name__ == "__main__":
    train()
