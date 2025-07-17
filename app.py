from flask import Flask, render_template, request
import os
import cv2
from ultralytics import YOLO
import easyocr
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# โหลดโมเดล
model = YOLO("runs/detect/train15/weights/best.pt")
reader = easyocr.Reader(['en'], gpu=True)

target_classes = ["0", "0R", "1", "1R", "2", "2R", "3", "3R", "4", "4R",
                  "5", "5R", "6", "6R", "7", "7R", "8", "8R", "9", "9R"]

def preprocess_gray_resize(image, scale=2):
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray

@app.route("/", methods=["GET", "POST"])
def index():
    result_text = ""
    uploaded_image_url = None
    result_image_url = None

    if request.method == "POST":
        file = request.files['image']
        if file:
            filename = f"{uuid.uuid4().hex}.jpg"
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            result_path = os.path.join(RESULT_FOLDER, filename)

            file.save(upload_path)
            uploaded_image_url = f"/{upload_path}"

            results = model(upload_path, device="cpu")
            original_image = cv2.imread(upload_path)

            # รวมกล่อง + class เฉพาะ target class
            boxes = results[0].boxes
            all_boxes = []
            for box, cls_id in zip(boxes.xyxy, boxes.cls.cpu().numpy().astype(int)):
                class_name = model.names[cls_id]
                if class_name in target_classes:
                    x1, y1, x2, y2 = map(int, box)
                    all_boxes.append(((x1, y1, x2, y2), class_name))

            # เรียงจากซ้ายไปขวา
            sorted_boxes = sorted(all_boxes, key=lambda b: b[0][0])

            digits = []

            for i, ((x1, y1, x2, y2), class_name) in enumerate(sorted_boxes):
                cropped = original_image[y1:y2, x1:x2]
                processed = preprocess_gray_resize(cropped)
                ocr_result = reader.readtext(processed, allowlist='0123456789', detail=0)
                num = ''.join(ocr_result).strip()
                if num:
                    digits.append(num)
                    # วาดกล่องและเลขบนภาพ
                    cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(original_image, num, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            result_text = ''.join(digits)
            cv2.imwrite(result_path, original_image)
            result_image_url = f"/{result_path}"

    return render_template("index.html",
                           result=result_text,
                           uploaded_image=uploaded_image_url,
                           result_image=result_image_url)

if __name__ == "__main__":
    app.run(debug=True)
