import cv2
import numpy as np
import pickle
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk

MODEL_PATH = "money_model.pkl"
IMAGE_SIZE = (200, 100)
DENOMS = ["10k", "20k", "50k", "100k", "200k", "500k"]

def _safe_resize(img, size):
    if img is None or img.size == 0: return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    try: return cv2.resize(img, size)
    except: return img

def preprocess_image(img):
    if img is None: return None
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    except: return img

def detect_money_box(img):
    """Returns (x, y, w, h) of the banknote or None."""
    if img is None: return None
    h_img, w_img = img.shape[:2]
    
    detect_width = 640
    scale = 1.0
    if w_img > detect_width:
        scale = detect_width / w_img
        work_img = cv2.resize(img, (detect_width, int(h_img * scale)))
    else:
        work_img = img
        
    h_work, w_work = work_img.shape[:2]
    min_area = h_work * w_work * 0.05
    
    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges_canny = cv2.Canny(blurred, 50, 150)
    
    thresh_adapt = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 5)
    
    best_box = None
    best_area = 0
    
    for binary_map in [thresh_otsu, edges_canny, thresh_adapt]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area: continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0
            if 1.2 < aspect_ratio < 3.5 and area > best_area:
                best_area = area
                best_box = (x, y, w, h)
                
    if best_box:
        x, y, w, h = best_box
        return (int(x/scale), int(y/scale), int(w/scale), int(h/scale))
    
    ch, cw = int(h_img * 0.8), int(w_img * 0.8)
    return ((w_img - cw)//2, (h_img - ch)//2, cw, ch)

def extract_color_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [36], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    cv2.normalize(hist_h, hist_h); cv2.normalize(hist_s, hist_s); cv2.normalize(hist_v, hist_v)
    
    hist_b = cv2.calcHist([img], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [32], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [32], [0, 256])
    cv2.normalize(hist_b, hist_b); cv2.normalize(hist_g, hist_g); cv2.normalize(hist_r, hist_r)
    
    moments = []
    for ch in cv2.split(hsv): moments.extend([np.mean(ch)/255.0, np.std(ch)/255.0])
    return np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten(), 
                           hist_b.flatten(), hist_g.flatten(), hist_r.flatten(), np.array(moments)])

def extract_texture_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    angle = cv2.phase(gx, gy, angleInDegrees=True)
    
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
        features.append(pooled1.flatten()); features.append(pooled2.flatten())
    return np.concatenate(features)

def extract_shape_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corner_count = np.sum(corners > 0.01 * corners.max())
    edges = cv2.Canny(gray, 50, 150)
    h, w = edges.shape
    regions = [edges[:h//2, :], edges[h//2:, :], edges[:, :w//2], edges[:, w//2:], edges[h//4:3*h//4, w//4:3*w//4]]
    edge_densities = [np.mean(r) / 255.0 for r in regions]
    return np.array([corner_count / 1000.0] + edge_densities)

def extract_all_features(img, size):
    img_resized = _safe_resize(img, size)
    color = extract_color_features(img_resized)
    texture = extract_texture_features(img_resized)
    cnn = extract_cnn_features(img_resized)
    shape = extract_shape_features(img_resized)
    features = np.concatenate([color, texture, cnn, shape])
    norm = np.linalg.norm(features)
    return features / norm if norm > 0 else features

def orb_match(img, templates, size):
    if not templates: return None, 0
    img = _safe_resize(img, size)
    orb = cv2.ORB_create(nfeatures=1000)
    kp, des = orb.detectAndCompute(img, None)
    if des is None or len(des) == 0: return None, 0
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_label, best_score = None, 0
    
    for label, temp_des in templates.items():
        if temp_des is None: continue
        matches = bf.match(des, temp_des)
        score = sum(1 for m in matches if m.distance < 50)
        if score > best_score:
            best_score = score
            best_label = label
    return best_label, best_score

class MoneyDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ Thống Nhận Diện Tiền Việt Nam")
        self.root.geometry("1000x600")
        self.root.configure(bg="#f0f0f0")
        
        self.model = self.load_model()
        
        self.create_sidebar()
        self.create_main_area()
        self.create_info_panel()
        
    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Lỗi", "Không tìm thấy file model! Vui lòng chạy train.py trước.")
            return None
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể load model: {e}")
            return None

    def create_sidebar(self):
        sidebar = tk.Frame(self.root, width=200, bg="#2c3e50")
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)
        
        tk.Label(sidebar, text="MENU", font=("Arial", 16, "bold"), bg="#2c3e50", fg="white").pack(pady=20)
        
        btn_style = {"font": ("Arial", 12), "bg": "#34495e", "fg": "white", "bd": 0, "pady": 10, "activebackground": "#1abc9c"}
        
        tk.Button(sidebar, text="📂 Chọn Ảnh", command=self.load_image, **btn_style).pack(fill=tk.X, pady=5, padx=10)
        tk.Button(sidebar, text="🔄 Reset", command=self.reset_ui, **btn_style).pack(fill=tk.X, pady=5, padx=10)
        tk.Button(sidebar, text="❌ Thoát", command=self.root.quit, **btn_style).pack(fill=tk.X, pady=5, padx=10)
        
        tk.Label(sidebar, text="Ver 2.1 (CNN+ORB)", font=("Arial", 8), bg="#2c3e50", fg="#95a5a6").pack(side=tk.BOTTOM, pady=10)

    def create_main_area(self):
        self.main_frame = tk.Frame(self.root, bg="#ecf0f1")
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(self.main_frame, text="Hình Ảnh Gốc / Nhận Diện", font=("Arial", 14, "bold"), bg="#ecf0f1").pack(pady=5)
        
        self.canvas_frame = tk.Frame(self.main_frame, bg="white", bd=2, relief=tk.SUNKEN)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.image_label = tk.Label(self.canvas_frame, text="Chưa có ảnh", bg="gray")
        self.image_label.pack(fill=tk.BOTH, expand=True)

    def create_info_panel(self):
        self.info_frame = tk.Frame(self.root, width=300, bg="white", bd=1, relief=tk.RAISED)
        self.info_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_frame.pack_propagate(False)
        
        tk.Label(self.info_frame, text="KẾT QUẢ PHÂN TÍCH", font=("Arial", 14, "bold"), bg="white", fg="#2c3e50").pack(pady=20)
        
        self.result_var = tk.StringVar(value="---")
        self.conf_var = tk.StringVar(value="0%")
        
        res_box = tk.Frame(self.info_frame, bg="#f7f9f9", pady=10)
        res_box.pack(fill=tk.X, padx=10)
        
        tk.Label(res_box, text="Mệnh giá dự đoán:", font=("Arial", 10), bg="#f7f9f9").pack()
        tk.Label(res_box, textvariable=self.result_var, font=("Arial", 24, "bold"), fg="#e74c3c", bg="#f7f9f9").pack()
        tk.Label(res_box, textvariable=self.conf_var, font=("Arial", 12), fg="#7f8c8d", bg="#f7f9f9").pack()

    def reset_ui(self):
        self.image_label.config(image='', text="Chưa có ảnh", bg="gray")
        self.result_var.set("---")
        self.conf_var.set("0%")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not path: return
        
        try:
            original_img = cv2.imread(path)
            if original_img is None: raise ValueError("Không đọc được ảnh")
            
            box = detect_money_box(original_img)
            x, y, w, h = box
            cropped = original_img[y:y+h, x:x+w]
            
            preprocessed = preprocess_image(cropped)
            features = extract_all_features(preprocessed, self.model['image_size'])
            
            if 'scaler' in self.model:
                features = self.model['scaler'].transform([features])[0]
            
            clf = self.model['clf']
            proba = clf.predict_proba([features])[0]
            classes = clf.classes_
            
            idx = np.argmax(proba)
            svm_label = classes[idx]
            svm_conf = proba[idx]
            
            orb_label, orb_score = orb_match(preprocessed, self.model['templates'], self.model['image_size'])
            
            final_label = svm_label
            final_conf = svm_conf
            
            if orb_label and orb_score > 25:
                if orb_label == svm_label:
                    final_conf = min(1.0, svm_conf + 0.1)
                elif orb_score > 40:
                    final_label = orb_label
                    final_conf = 0.9
            
            self.result_var.set(final_label)
            self.conf_var.set(f"{final_conf*100:.1f}%")
            
            display_img = original_img.copy()
            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(display_img, f"{final_label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            h_disp, w_disp = display_img.shape[:2]
            
            canvas_h = self.canvas_frame.winfo_height()
            canvas_w = self.canvas_frame.winfo_width()
            if canvas_h > 100 and canvas_w > 100:
                scale = min(canvas_w/w_disp, canvas_h/h_disp)
                new_size = (int(w_disp*scale), int(h_disp*scale))
                display_img = cv2.resize(display_img, new_size)
            
            img_pil = Image.fromarray(display_img)
            img_tk = ImageTk.PhotoImage(img_pil)
            self.image_label.config(image=img_tk, text="")
            self.image_label.image = img_tk
            
        except Exception as e:
            messagebox.showerror("Lỗi xử lý", str(e))
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = MoneyDashboard(root)
    root.mainloop()
