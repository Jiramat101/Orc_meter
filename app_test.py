# import os
# import uuid
# import random
# import numpy as np
# import cv2
# import torch

# from flask import Flask, render_template, request
# from ultralytics import YOLO
# import easyocr
# from collections import Counter

# # ---------- Deterministic settings (ให้เหมือนตอนรัน console) ----------
# SEED = 1234
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# cv2.setNumThreads(1)

# # ---------- Flask setup ----------
# app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # ป้องกันเบราเซอร์ส่งไฟล์ใหญ่เกิน
# UPLOAD_FOLDER = 'static/uploads'
# RESULT_FOLDER = 'static/results'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULT_FOLDER, exist_ok=True)

# # ---------- Device & models ----------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model = YOLO("runs/detect/train15/weights/best.pt")
# reader = easyocr.Reader(['en'], gpu=(DEVICE == "cuda"))

# # ถ้าชื่อคลาสในโมเดลเป็นแยกตัวเลขกับ R ให้คงไว้เหมือนเดิม
# target_classes = ["0", "0R", "1", "1R", "2", "2R", "3", "3R", "4", "4R",
#                   "5", "5R", "6", "6R", "7", "7R", "8", "8R", "9", "9R"]

# # ---------- Preprocess: ยกจาก console มาแบบเข้มข้น ----------
# def preprocess_for_ocr(bgr, scale=3, invert=False, use_clahe=True, sharpen=True):
#     # Resize ก่อน (คอนโซลมักจะขยายภาพก่อน OCR)
#     img = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Sharpen เพิ่มความคม (เหมือนที่มักทำในสคริปต์)
#     if sharpen:
#         kernel = np.array([[0, -1, 0],
#                            [-1, 5, -1],
#                            [0, -1, 0]])
#         gray = cv2.filter2D(gray, -1, kernel)

#     # เพิ่มคอนทราสต์ด้วย CLAHE
#     if use_clahe:
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         gray = clahe.apply(gray)

#     # ลด noise เบา ๆ
#     gray = cv2.GaussianBlur(gray, (3, 3), 0)

#     # Threshold แบบ Otsu
#     _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # กรณีพื้นหลังเข้ม-ตัวเลขสว่าง บางภาพต้อง invert
#     if invert:
#         th = 255 - th

#     return th

# # Post-fix ความสับสนตัวอักษร/เลข
# def fix_common_ocr_errors(text):
#     mapping = {"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "s": "5", "B": "8"}
#     return ''.join(mapping.get(ch, ch) for ch in text)

# # อ่านเลขด้วยการลองหลายพรีเซต (เหมือนเราทดลองหลายวิธีในคอนโซลแล้วเลือกคำตอบที่ซ้ำกันมากสุด)
# def robust_read_digits(cropped_bgr):
#     candidates = []

#     # ชุดพารามิเตอร์ที่ “คัฟเวอร์” เคสหลัก ๆ
#     param_sets = [
#         dict(scale=2, invert=False, use_clahe=True,  sharpen=True),
#         dict(scale=3, invert=False, use_clahe=True,  sharpen=True),
#         dict(scale=4, invert=False, use_clahe=True,  sharpen=True),
#         dict(scale=3, invert=True,  use_clahe=True,  sharpen=True),
#         dict(scale=3, invert=False, use_clahe=False, sharpen=True),
#         dict(scale=3, invert=False, use_clahe=True,  sharpen=False),
#     ]

#     for ps in param_sets:
#         processed = preprocess_for_ocr(cropped_bgr, **ps)
#         # อ่านเฉพาะตัวเลข
#         out = reader.readtext(processed, allowlist='0123456789', detail=0, paragraph=False)
#         if not out:
#             continue
#         num = ''.join(out).strip()
#         num = fix_common_ocr_errors(num)
#         # เก็บเฉพาะที่เป็นตัวเลขล้วน
#         if num and all(c.isdigit() for c in num):
#             candidates.append(num)

#     if not candidates:
#         return ""

#     # เลือกผลที่เจอบ่อยสุด (เสียงข้างมาก)
#     most_common = Counter(candidates).most_common(1)[0][0]
#     return most_common

# @app.route("/", methods=["GET", "POST"])
# def index():
#     result_text = ""
#     uploaded_image_url = None
#     result_image_url = None
#     debug_info = []  # optional: ไว้โชว์/ตรวจสอบ

#     if request.method == "POST" and 'image' in request.files:
#         f = request.files['image']

#         # เซฟ “ไฟล์ดิบ” ที่ browser ส่งมา โดยไม่แปลง/บีบอัดซ้ำ
#         # (สำคัญมากเพื่อให้เหมือน console)
#         ext = os.path.splitext(f.filename)[1].lower() or ".jpg"
#         filename = f"{uuid.uuid4().hex}{ext}"
#         upload_path = os.path.join(UPLOAD_FOLDER, filename)
#         result_path = os.path.join(RESULT_FOLDER, filename)

#         # บันทึกด้วย byte stream ตรง ๆ
#         f.stream.seek(0)
#         raw = f.stream.read()
#         with open(upload_path, "wb") as w:
#             w.write(raw)

#         uploaded_image_url = f"/{upload_path}"

#         # โหลดภาพต้นฉบับจากดิสก์ (เหมือน console ทำ cv2.imread จากไฟล์)
#         original_image = cv2.imread(upload_path, cv2.IMREAD_COLOR)

#         # รัน YOLO ด้วย conf ที่ค่อนข้างต่ำเพื่อไม่พลาดตัวเล็ก ๆ (ปรับตามงานจริงได้)
#         results = model(upload_path, device=DEVICE, conf=0.5, iou=0.3, verbose=False)

#         # รวมกล่องเฉพาะคลาสเป้าหมาย
#         boxes = results[0].boxes
#         all_boxes = []
#         for box, cls_id in zip(boxes.xyxy, boxes.cls.cpu().numpy().astype(int)):
#             class_name = model.names[cls_id]
#             if class_name in target_classes:
#                 x1, y1, x2, y2 = map(int, box)
#                 # กันเลยขอบภาพ
#                 x1 = max(0, x1); y1 = max(0, y1)
#                 x2 = min(original_image.shape[1]-1, x2)
#                 y2 = min(original_image.shape[0]-1, y2)
#                 if x2 > x1 and y2 > y1:
#                     all_boxes.append(((x1, y1, x2, y2), class_name))

#         # เรียงซ้าย->ขวา
#         sorted_boxes = sorted(all_boxes, key=lambda b: b[0][0])

#         digits = []
#         for (x1, y1, x2, y2), cname in sorted_boxes:
#             crop = original_image[y1:y2, x1:x2].copy()
#             num = robust_read_digits(crop)

#             # วาดผลลงภาพ
#             color = (0, 255, 0) if num else (0, 0, 255)
#             cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(original_image, num if num else "?", (x1, max(0, y1 - 8)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

#             if num:
#                 digits.append(num)

#         result_text = ''.join(digits)

#         # เซฟผลลัพธ์
#         cv2.imwrite(result_path, original_image)
#         result_image_url = f"/{result_path}"

#     return render_template(
#         "index.html",
#         result=result_text,
#         uploaded_image=uploaded_image_url,
#         result_image=result_image_url
#     )

# if __name__ == "__main__":
#     # ใช้ debug=True เฉพาะตอนพัฒนา
#     app.run(host="0.0.0.0", port=5000, debug=True)

# import os
# from flask import Flask, render_template, request, send_from_directory
# from ultralytics import YOLO
# import cv2
# import easyocr
# import torch
# from PIL import Image

# # ตั้งค่าโฟลเดอร์
# UPLOAD_FOLDER = 'static/uploads'
# RESULT_FOLDER = 'static/results'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULT_FOLDER, exist_ok=True)

# # โหลดโมเดล YOLO
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model = YOLO("runs/detect/train15/weights/best.pt")

# # โหลด EasyOCR
# reader = easyocr.Reader(['en'], gpu=(DEVICE == "cuda"))

# app = Flask(__name__)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     uploaded_images = []
#     result_images = []
#     results_text = []

#     if request.method == 'POST':
#         files = request.files.getlist("images")
#         for file in files:
#             if file and file.filename != '':
#                 # บันทึกไฟล์อัปโหลด
#                 filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#                 file.save(filepath)
#                 uploaded_images.append(filepath)

#                 # ตรวจจับด้วย YOLO
#                 results = model(filepath, device=DEVICE, conf=0.25, iou=0.45, verbose=False)
#                 result_img_path = os.path.join(RESULT_FOLDER, f"result_{file.filename}")
#                 results[0].save(result_img_path)

#                 # อ่านตัวเลขด้วย EasyOCR
#                 img_cv2 = cv2.imread(filepath)
#                 ocr_result = reader.readtext(img_cv2, detail=0)
#                 detected_text = " ".join(ocr_result)
#                 results_text.append(detected_text)
#                 result_images.append(result_img_path)

#     return render_template('index.html',
#                            uploaded_images=uploaded_images,
#                            result_images=result_images,
#                            results_text=results_text)


# @app.route('/static/<path:path>')
# def send_static(path):
#     return send_from_directory('static', path)


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)


# import os
# import uuid
# import random
# import numpy as np
# import cv2
# import torch

# from flask import Flask, render_template, request
# from ultralytics import YOLO
# import easyocr
# from collections import Counter

# # ---------- Deterministic settings ----------
# SEED = 1234
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# cv2.setNumThreads(1)

# # ---------- Flask setup ----------
# app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
# UPLOAD_FOLDER = 'static/uploads'
# RESULT_FOLDER = 'static/results'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULT_FOLDER, exist_ok=True)

# # ---------- Device & models ----------
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# model = YOLO("runs/detect/train15/weights/best.pt")
# reader = easyocr.Reader(['en'], gpu=(DEVICE == "cuda"))

# target_classes = ["0", "0R", "1", "1R", "2", "2R", "3", "3R", "4", "4R",
#                   "5", "5R", "6", "6R", "7", "7R", "8", "8R", "9", "9R"]

# # ---------- Preprocessing ----------
# def preprocess_for_ocr(bgr, scale=3, invert=False, use_clahe=True, sharpen=True):
#     img = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     if sharpen:
#         kernel = np.array([[0, -1, 0],
#                            [-1, 5, -1],
#                            [0, -1, 0]])
#         gray = cv2.filter2D(gray, -1, kernel)
#     if use_clahe:
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         gray = clahe.apply(gray)
#     gray = cv2.GaussianBlur(gray, (3, 3), 0)
#     _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     if invert:
#         th = 255 - th
#     return th

# def fix_common_ocr_errors(text):
#     mapping = {"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "s": "5", "B": "8"}
#     return ''.join(mapping.get(ch, ch) for ch in text)

# def robust_read_digits(cropped_bgr):
#     candidates = []
#     param_sets = [
#         dict(scale=2, invert=False, use_clahe=True,  sharpen=True),
#         dict(scale=3, invert=False, use_clahe=True,  sharpen=True),
#         dict(scale=4, invert=False, use_clahe=True,  sharpen=True),
#         dict(scale=3, invert=True,  use_clahe=True,  sharpen=True),
#         dict(scale=3, invert=False, use_clahe=False, sharpen=True),
#         dict(scale=3, invert=False, use_clahe=True,  sharpen=False),
#     ]
#     for ps in param_sets:
#         processed = preprocess_for_ocr(cropped_bgr, **ps)
#         out = reader.readtext(processed, allowlist='0123456789', detail=0, paragraph=False)
#         if not out:
#             continue
#         num = ''.join(out).strip()
#         num = fix_common_ocr_errors(num)
#         if num and all(c.isdigit() for c in num):
#             candidates.append(num)
#     if not candidates:
#         return ""
#     return Counter(candidates).most_common(1)[0][0]

# @app.route("/", methods=["GET", "POST"])
# def index():
#     results_data = []

#     if request.method == "POST":
#         files = request.files.getlist("images")
#         for f in files:
#             if not f.filename:
#                 continue

#             ext = os.path.splitext(f.filename)[1].lower() or ".jpg"
#             filename = f"{uuid.uuid4().hex}{ext}"
#             upload_path = os.path.join(UPLOAD_FOLDER, filename)
#             result_path = os.path.join(RESULT_FOLDER, filename)

#             # Save original image
#             f.save(upload_path)

#             # Load image
#             original_image = cv2.imread(upload_path, cv2.IMREAD_COLOR)

#             # Run YOLO
#             results = model(upload_path, device=DEVICE, conf=0.25, iou=0.45, verbose=False)
#             boxes = results[0].boxes
#             all_boxes = []
#             for box, cls_id in zip(boxes.xyxy, boxes.cls.cpu().numpy().astype(int)):
#                 class_name = model.names[cls_id]
#                 if class_name in target_classes:
#                     x1, y1, x2, y2 = map(int, box)
#                     x1 = max(0, x1); y1 = max(0, y1)
#                     x2 = min(original_image.shape[1]-1, x2)
#                     y2 = min(original_image.shape[0]-1, y2)
#                     if x2 > x1 and y2 > y1:
#                         all_boxes.append(((x1, y1, x2, y2), class_name))

#             sorted_boxes = sorted(all_boxes, key=lambda b: b[0][0])

#             digits = []
#             for (x1, y1, x2, y2), cname in sorted_boxes:
#                 crop = original_image[y1:y2, x1:x2].copy()
#                 num = robust_read_digits(crop)
#                 color = (0, 255, 0) if num else (0, 0, 255)
#                 cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(original_image, num if num else "?", (x1, max(0, y1 - 8)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
#                 if num:
#                     digits.append(num)

#             result_text = ''.join(digits)
#             cv2.imwrite(result_path, original_image)

#             results_data.append({
#                 "original": f"/{upload_path}",
#                 "result_img": f"/{result_path}",
#                 "digits": result_text
#             })

#     return render_template("index.html", results=results_data)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)




import os 
import uuid
import random
import numpy as np
import cv2
import torch

from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import easyocr
from collections import Counter
from io import BytesIO
import pandas as pd  # << เพิ่ม
from datetime import datetime

# ---------- Deterministic settings ----------
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
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ---------- Device & models ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("runs/detect/train15/weights/best.pt")
reader = easyocr.Reader(['en'], gpu=(DEVICE == "cuda"))

target_classes = ["0", "0R", "1", "1R", "2", "2R", "3", "3R", "4", "4R",
                  "5", "5R", "6", "6R", "7", "7R", "8", "8R", "9", "9R"]

# ---------- memory เก็บผลลัพธ์รอบล่าสุด สำหรับ export ----------
LAST_RESULTS = []  # list ของ dict: {"original": "/static/uploads/...", "result_img": "/static/results/...", "digits": "..."}

# ---------- Preprocessing ----------
def preprocess_for_ocr(bgr, scale=3, invert=False, use_clahe=True, sharpen=True):
    img = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if sharpen:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert:
        th = 255 - th
    return th

def fix_common_ocr_errors(text):
    mapping = {"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "s": "5", "B": "8"}
    return ''.join(mapping.get(ch, ch) for ch in text)

def robust_read_digits(cropped_bgr):
    candidates = []
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
        out = reader.readtext(processed, allowlist='0123456789', detail=0, paragraph=False)
        if not out:
            continue
        num = ''.join(out).strip()
        num = fix_common_ocr_errors(num)
        if num and all(c.isdigit() for c in num):
            candidates.append(num)
    if not candidates:
        return ""
    return Counter(candidates).most_common(1)[0][0]

@app.route("/", methods=["GET", "POST"])
def index():
    global LAST_RESULTS
    results_data = []

    if request.method == "POST":
        files = request.files.getlist("images")
        for f in files:
            if not f.filename:
                continue

            ext = os.path.splitext(f.filename)[1].lower() or ".jpg"
            filename = f"{uuid.uuid4().hex}{ext}"
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            result_path = os.path.join(RESULT_FOLDER, filename)

            # Save original image
            f.save(upload_path)

            # Load image
            original_image = cv2.imread(upload_path, cv2.IMREAD_COLOR)

            # Run YOLO (ค่าหย่อนมือให้เหมือนเวอร์ชันที่แม่นกว่า)
            results = model(upload_path, device=DEVICE, conf=0.25, iou=0.45, verbose=False)
            boxes = results[0].boxes
            all_boxes = []
            for box, cls_id in zip(boxes.xyxy, boxes.cls.cpu().numpy().astype(int)):
                class_name = model.names[cls_id]
                if class_name in target_classes:
                    x1, y1, x2, y2 = map(int, box)
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(original_image.shape[1]-1, x2)
                    y2 = min(original_image.shape[0]-1, y2)
                    if x2 > x1 and y2 > y1:
                        all_boxes.append(((x1, y1, x2, y2), class_name))

            # sort L->R
            sorted_boxes = sorted(all_boxes, key=lambda b: b[0][0])

            digits = []
            for (x1, y1, x2, y2), _ in sorted_boxes:
                crop = original_image[y1:y2, x1:x2].copy()
                num = robust_read_digits(crop)
                color = (0, 255, 0) if num else (0, 0, 255)
                cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(original_image, num if num else "?", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
                if num:
                    digits.append(num)

            result_text = ''.join(digits)
            cv2.imwrite(result_path, original_image)

            results_data.append({
                "original": f"/{upload_path}",
                "result_img": f"/{result_path}",
                "digits": result_text,
                "source_name": f.filename
            })

    # เก็บผลล่าสุดไว้สำหรับ export
    if request.method == "POST":
        LAST_RESULTS = results_data

    return render_template("index.html", results=results_data)

@app.route("/export", methods=["POST"])
def export():
    """ส่งออกผลลัพธ์ล่าสุดเป็น Excel ให้ผู้ใช้ตั้งชื่อไฟล์ได้"""
    # รับชื่อไฟล์จากฟอร์ม
    filename = request.form.get("export_filename", "").strip()
    if not filename:
        # ค่าดีฟอลต์แนบ timestamp กันชนกัน
        filename = f"digits_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    if not filename.lower().endswith(".xlsx"):
        filename += ".xlsx"

    # หากยังไม่มีผลลัพธ์
    if not LAST_RESULTS:
        # สร้างไฟล์เปล่าอย่างสุภาพ
        df_empty = pd.DataFrame(columns=["No.", "Digits", "Original Image", "Result Image", "Source Name"])
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df_empty.to_excel(writer, index=False, sheet_name="Results")
        bio.seek(0)
        return send_file(
            bio,
            as_attachment=True,
            download_name=filename,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # แปลงผลลัพธ์ล่าสุดเป็นตาราง
    rows = []
    for idx, item in enumerate(LAST_RESULTS, start=1):
        rows.append({
            "No.": idx,
            "Digits": item.get("digits", ""),
            "Original Image": item.get("original", ""),
            "Result Image": item.get("result_img", ""),
            "Source Name": item.get("source_name", "")
        })

    df = pd.DataFrame(rows, columns=["No.", "Digits", "Original Image", "Result Image", "Source Name"])

    # เขียนเป็นไฟล์ Excel ในหน่วยความจำแล้วส่งคืนให้ดาวน์โหลด
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
