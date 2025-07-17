from ultralytics import YOLO
import cv2
import os
import glob
import tkinter as tk
from tkinter import filedialog

# --- GUI ให้ผู้ใช้เลือกโฟลเดอร์ ---
root = tk.Tk()
root.withdraw()
input_folder = filedialog.askdirectory(title="เลือกโฟลเดอร์ภาพต้นฉบับ")

if not input_folder:
    print("❌ ไม่ได้เลือกโฟลเดอร์")
    exit()

# โหลดโมเดล YOLOv8
model = YOLO("runs/detect/train6/weights/best.pt")  # เปลี่ยนตามโมเดลที่คุณเทรนไว้

# รายชื่อ class (เรียงตรงกับโมเดล)
class_names = ["room", "use"]

# โฟลเดอร์สำหรับภาพที่ครอป
crop_dirs = {
    "room": os.path.join(input_folder, "crops_room"),
    "use": os.path.join(input_folder, "crops_use")
}
for folder in crop_dirs.values():
    os.makedirs(folder, exist_ok=True)

# ค้นหารูป .jpg และ .png ทั้งหมด
image_paths = glob.glob(os.path.join(input_folder, "*.jpg")) + \
              glob.glob(os.path.join(input_folder, "*.png"))

if not image_paths:
    print("❌ ไม่พบภาพในโฟลเดอร์")
    exit()

# ประมวลผลแต่ละภาพ
for img_path in image_paths:
    img = cv2.imread(img_path)
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    results = model(img)

    for result in results:
        boxes = result.boxes

        for j, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            if cls_id >= len(class_names):
                continue

            cls_name = class_names[cls_id]
            if cls_name not in crop_dirs:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_img = img[y1:y2, x1:x2]

            # ชื่อไฟล์ (ไม่มีนามสกุล)
            base_filename = f"{img_name}_{cls_name}_{j:03}"

            # พาธสำหรับ .jpg และ .txt
            img_save_path = os.path.join(crop_dirs[cls_name], base_filename + ".jpg")
            txt_save_path = os.path.join(crop_dirs[cls_name], base_filename + ".txt")

            # บันทึกภาพครอป
            cv2.imwrite(img_save_path, cropped_img)

            # สร้างไฟล์ .txt ว่าง
            with open(txt_save_path, 'w') as f:
                pass  # ไฟล์ว่าง

            print(f"✅ Saved: {img_save_path} + {txt_save_path}")
