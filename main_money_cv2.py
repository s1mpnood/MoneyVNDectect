import cv2
import numpy as np
import os

TEMPLATE_DIR = "templates"  # thư mục chứa ảnh mẫu

# ================== HÀM HỖ TRỢ ==================

def load_templates(template_dir=TEMPLATE_DIR):
    """
    Đọc tất cả ảnh trong thư mục templates/,
    trả về dict: label -> BGR image
    """
    templates = {}
    for fname in os.listdir(template_dir):
        path = os.path.join(template_dir, fname)
        if not os.path.isfile(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        label = os.path.splitext(fname)[0]  # "10k.jpg" -> "10k"
        templates[label] = img
    return templates

# ---------- 1. ORB FEATURE MATCHING ----------

def compute_orb_features(img_gray):
    orb = cv2.ORB_create(1000)
    kp, des = orb.detectAndCompute(img_gray, None)
    return kp, des

def score_orb(roi, templates):
    """
    Tính điểm ORB cho mỗi template, chọn label có score cao nhất.
    """
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    kp_roi, des_roi = compute_orb_features(gray_roi)
    if des_roi is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    best_label = None
    best_score = 0

    for label, tmpl in templates.items():
        gray_tmpl = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
        kp_t, des_t = compute_orb_features(gray_tmpl)
        if des_t is None:
            continue
        matches = bf.match(des_roi, des_t)
        # sort theo distance tăng dần
        matches = sorted(matches, key=lambda x: x.distance)
        # chọn các match tốt (distance nhỏ)
        good = [m for m in matches if m.distance < 60]
        score = len(good)
        if score > best_score:
            best_score = score
            best_label = label

    return best_label

# ---------- 2. HISTOGRAM HSV ----------

def compute_hist_hs(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None,
                        [50, 60], [0, 180, 0, 256])  # H:50 bins, S:60 bins
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def score_histogram(roi, templates):
    """
    So sánh histogram HSV giữa ROI và từng template.
    Dùng compareHist với phương pháp CORREL, chọn highest.
    """
    hist_roi = compute_hist_hs(roi)
    best_label = None
    best_score = -1  # vì CORREL: -1 đến 1

    for label, tmpl in templates.items():
        hist_t = compute_hist_hs(tmpl)
        score = cv2.compareHist(hist_roi, hist_t, cv2.HISTCMP_CORREL)
        if score > best_score:
            best_score = score
            best_label = label

    return best_label

# ---------- 3. TEMPLATE MATCHING ----------

def score_template_matching(roi, templates):
    """
    Resize ROI gần với kích thước template, dùng matchTemplate.
    Chọn label có max_val lớn nhất.
    """
    best_label = None
    best_val = -1

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    for label, tmpl in templates.items():
        tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)

        # Resize ROI về kích thước template (hoặc ngược lại)
        h_t, w_t = tmpl_gray.shape[:2]
        roi_resized = cv2.resize(roi_gray, (w_t, h_t))

        # Template Matching
        res = cv2.matchTemplate(roi_resized, tmpl_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val > best_val:
            best_val = max_val
            best_label = label

    return best_label

# ---------- TÌM TỜ TIỀN BẰNG CONTOUR ----------

def detect_note_contour(frame):
    """
    Tìm contour lớn nhất dạng gần hình chữ nhật,
    cắt ra ROI tờ tiền. Nếu không thấy thì trả về None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # Chọn contour lớn nhất
    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    x, y, w, h = cv2.boundingRect(c)

    # Nếu diện tích quá nhỏ, bỏ qua
    H, W = frame.shape[:2]
    if w * h < 0.05 * W * H:
        return None, None

    # Cắt ROI (tạm dùng bounding box cho đơn giản)
    roi = frame[y:y+h, x:x+w]
    return roi, (x, y, w, h)

# ---------- KẾT HỢP 3 THUẬT TOÁN ----------

def vote_result(roi, templates):
    """
    Chạy 3 thuật toán, vote label cuối cùng.
    """
    if roi is None:
        return None, {}

    label_orb = score_orb(roi, templates)
    label_hist = score_histogram(roi, templates)
    label_tmpl = score_template_matching(roi, templates)

    votes = {}
    for l in [label_orb, label_hist, label_tmpl]:
        if l is None:
            continue
        votes[l] = votes.get(l, 0) + 1

    if votes:
        final_label = max(votes.items(), key=lambda x: x[1])[0]
    else:
        final_label = None

    details = {
        "ORB": label_orb,
        "HIST": label_hist,
        "TMPL": label_tmpl
    }
    return final_label, details

# ================== MAIN: WEBCAM ==================

def main():
    templates = load_templates()
    if not templates:
        print("Không tìm thấy template nào trong thư mục 'templates/'.")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không mở được webcam.")
        return

    print("Bấm 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi, bbox = detect_note_contour(frame)

        if roi is not None and bbox is not None:
            (x, y, w, h) = bbox

            final_label, details = vote_result(roi, templates)

            # Vẽ bounding box
            color_box = (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color_box, 2)

            # Ghi kết quả
            text = f"Guess: {final_label}" if final_label else "Guess: ?"
            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # (Optional) ghi thêm chi tiết từng thuật toán
            debug_text = f"O:{details['ORB']} H:{details['HIST']} T:{details['TMPL']}"
            cv2.putText(frame, debug_text, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("VN Money - cv2 only", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
