from ultralytics import YOLO 
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from collections import Counter
import easyocr
import numpy as np

# ฟังก์ชัน preprocess: grayscale + resize
def preprocess_gray_resize(image, scale=2):
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray

# ปิดหน้าต่างหลักของ Tkinter
Tk().withdraw()

# เปิด File Dialog ให้ผู้ใช้เลือกภาพ
image_path = askopenfilename(
    title="เลือกรูปภาพที่ต้องการตรวจจับ",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
)

if not image_path:
    print("ไม่ได้เลือกรูปภาพใด ๆ")
else:
    # โหลดโมเดล YOLOv8 และใช้ CPU เท่านั้น
    model = YOLO("runs/detect/train15/weights/best.pt")
    results = model(image_path, device="cpu")
    boxes = results[0].boxes

    # นับจำนวนแต่ละ class ที่เจอ
    if boxes is not None and boxes.cls is not None:
        class_ids = boxes.cls.cpu().numpy().astype(int)
        class_names = [model.names[i] for i in class_ids]
        counts = Counter(class_names)

        print("พบวัตถุดังนี้ในภาพ:")
        for name, count in counts.items():
            print(f"- {name}: {count} ชิ้น")
    else:
        print("ไม่พบวัตถุใด ๆ")

    # แสดงภาพผลลัพธ์
    annotated_frame = results[0].plot()
    cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # OCR เฉพาะ class ที่ต้องการ
    target_classes = ["0", "0R", "1", "1R", "2", "2R", "3", "3R", "4", "4R", "5", "5R", "6", "6R", "7", "7R", "8", "8R", "9", "9R"]
    original_image = cv2.imread(image_path)

    print("\n📌 ผลลัพธ์ OCR (เฉพาะ class ที่กำหนด):")

    reader = easyocr.Reader(['en'], gpu=True)

    # 🟢 รวบรวมกล่องและ class name เฉพาะ target class
    all_boxes = []
    for i, (box, cls_id) in enumerate(zip(boxes.xyxy, boxes.cls.cpu().numpy().astype(int))):
        class_name = model.names[cls_id]
        if class_name in target_classes:
            x1, y1, x2, y2 = map(int, box)
            all_boxes.append(((x1, y1, x2, y2), class_name))

    # 🔃 เรียงกล่องจากซ้าย → ขวา (ตาม x1)
    sorted_boxes = sorted(all_boxes, key=lambda b: b[0][0])  # b[0][0] = x1

    # 🧠 OCR ตามลำดับเรียงแล้ว
    for i, ((x1, y1, x2, y2), class_name) in enumerate(sorted_boxes):
        cropped = original_image[y1:y2, x1:x2]
        processed = preprocess_gray_resize(cropped)
        ocr_result = reader.readtext(processed, allowlist='0123456789', detail=0)
        text = ''.join(ocr_result).strip()

        print(f"\n[class: {class_name}] กล่อง {i+1}:")
        print(f"  ➤ อ่านได้ว่า: {text}")

        cv2.imshow(f"{class_name} #{i+1}", processed)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
