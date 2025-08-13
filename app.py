import os
import uuid
import random
import numpy as np
import cv2
import torch

from flask import Flask, render_template, request
from ultralytics import YOLO
import easyocr
from collections import Counter

# ---------- Deterministic settings (ให้เหมือนตอนรัน console) ----------
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
cv2.setNumThreads(1)

# ---------- Flask setup ----------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # ป้องกันเบราเซอร์ส่งไฟล์ใหญ่เกิน
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ---------- Device & models ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("runs/detect/train15/weights/best.pt")
reader = easyocr.Reader(['en'], gpu=(DEVICE == "cuda"))

# ถ้าชื่อคลาสในโมเดลเป็นแยกตัวเลขกับ R ให้คงไว้เหมือนเดิม
target_classes = ["0", "0R", "1", "1R", "2", "2R", "3", "3R", "4", "4R",
                  "5", "5R", "6", "6R", "7", "7R", "8", "8R", "9", "9R"]

# ---------- Preprocess: ยกจาก console มาแบบเข้มข้น ----------
def preprocess_for_ocr(bgr, scale=3, invert=False, use_clahe=True, sharpen=True):
    # Resize ก่อน (คอนโซลมักจะขยายภาพก่อน OCR)
    img = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sharpen เพิ่มความคม (เหมือนที่มักทำในสคริปต์)
    if sharpen:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)

    # เพิ่มคอนทราสต์ด้วย CLAHE
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # ลด noise เบา ๆ
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Threshold แบบ Otsu
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # กรณีพื้นหลังเข้ม-ตัวเลขสว่าง บางภาพต้อง invert
    if invert:
        th = 255 - th

    return th

# Post-fix ความสับสนตัวอักษร/เลข
def fix_common_ocr_errors(text):
    mapping = {"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "s": "5", "B": "8"}
    return ''.join(mapping.get(ch, ch) for ch in text)

# อ่านเลขด้วยการลองหลายพรีเซต (เหมือนเราทดลองหลายวิธีในคอนโซลแล้วเลือกคำตอบที่ซ้ำกันมากสุด)
def robust_read_digits(cropped_bgr):
    candidates = []

    # ชุดพารามิเตอร์ที่ “คัฟเวอร์” เคสหลัก ๆ
    param_sets = [
        dict(scale=2, invert=False, use_clahe=True,  sharpen=True),
        dict(scale=3, invert=False, use_clahe=True,  sharpen=True),
        dict(scale=4, invert=False, use_clahe=True,  sharpen=True),
        dict(scale=3, invert=True,  use_clahe=True,  sharpen=True),
        dict(scale=3, invert=False, use_clahe=False, sharpen=True),
        dict(scale=3, invert=False, use_clahe=True,  sharpen=False),
    ]

    for ps in param_sets:
        processed = preprocess_for_ocr(cropped_bgr, **ps)
        # อ่านเฉพาะตัวเลข
        out = reader.readtext(processed, allowlist='0123456789', detail=0, paragraph=False)
        if not out:
            continue
        num = ''.join(out).strip()
        num = fix_common_ocr_errors(num)
        # เก็บเฉพาะที่เป็นตัวเลขล้วน
        if num and all(c.isdigit() for c in num):
            candidates.append(num)

    if not candidates:
        return ""

    # เลือกผลที่เจอบ่อยสุด (เสียงข้างมาก)
    most_common = Counter(candidates).most_common(1)[0][0]
    return most_common

@app.route("/", methods=["GET", "POST"])
def index():
    result_text = ""
    uploaded_image_url = None
    result_image_url = None
    debug_info = []  # optional: ไว้โชว์/ตรวจสอบ

    if request.method == "POST" and 'image' in request.files:
        f = request.files['image']

        # เซฟ “ไฟล์ดิบ” ที่ browser ส่งมา โดยไม่แปลง/บีบอัดซ้ำ
        # (สำคัญมากเพื่อให้เหมือน console)
        ext = os.path.splitext(f.filename)[1].lower() or ".jpg"
        filename = f"{uuid.uuid4().hex}{ext}"
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        result_path = os.path.join(RESULT_FOLDER, filename)

        # บันทึกด้วย byte stream ตรง ๆ
        f.stream.seek(0)
        raw = f.stream.read()
        with open(upload_path, "wb") as w:
            w.write(raw)

        uploaded_image_url = f"/{upload_path}"

        # โหลดภาพต้นฉบับจากดิสก์ (เหมือน console ทำ cv2.imread จากไฟล์)
        original_image = cv2.imread(upload_path, cv2.IMREAD_COLOR)

        # รัน YOLO ด้วย conf ที่ค่อนข้างต่ำเพื่อไม่พลาดตัวเล็ก ๆ (ปรับตามงานจริงได้)
        results = model(upload_path, device=DEVICE, conf=0.5, iou=0.3, verbose=False)

        # รวมกล่องเฉพาะคลาสเป้าหมาย
        boxes = results[0].boxes
        all_boxes = []
        for box, cls_id in zip(boxes.xyxy, boxes.cls.cpu().numpy().astype(int)):
            class_name = model.names[cls_id]
            if class_name in target_classes:
                x1, y1, x2, y2 = map(int, box)
                # กันเลยขอบภาพ
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(original_image.shape[1]-1, x2)
                y2 = min(original_image.shape[0]-1, y2)
                if x2 > x1 and y2 > y1:
                    all_boxes.append(((x1, y1, x2, y2), class_name))

        # เรียงซ้าย->ขวา
        sorted_boxes = sorted(all_boxes, key=lambda b: b[0][0])

        digits = []
        for (x1, y1, x2, y2), cname in sorted_boxes:
            crop = original_image[y1:y2, x1:x2].copy()
            num = robust_read_digits(crop)

            # วาดผลลงภาพ
            color = (0, 255, 0) if num else (0, 0, 255)
            cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(original_image, num if num else "?", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

            if num:
                digits.append(num)

        result_text = ''.join(digits)

        # เซฟผลลัพธ์
        cv2.imwrite(result_path, original_image)
        result_image_url = f"/{result_path}"

    return render_template(
        "index.html",
        result=result_text,
        uploaded_image=uploaded_image_url,
        result_image=result_image_url
    )

if __name__ == "__main__":
    # ใช้ debug=True เฉพาะตอนพัฒนา
    app.run(host="0.0.0.0", port=5000, debug=True)
