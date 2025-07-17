import cv2
from craft_text_detector import Craft
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import easyocr
import numpy as np

# เปิดหน้าต่างเลือกไฟล์ภาพ
Tk().withdraw()
image_path = askopenfilename(
    title="เลือกรูปภาพที่ต้องการตรวจจับ",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
)

if not image_path:
    print("ไม่ได้เลือกรูปภาพใด ๆ")
    exit()

image = cv2.imread(image_path)
if image is None:
    print("โหลดรูปภาพไม่สำเร็จ")
    exit()

# สร้าง craft detector
craft = Craft(output_dir=None, cuda=False)

try:
    boxes, polys, score_text = craft.detect_text(image)
except Exception as e:
    print("Error ในการ detect text:", e)
    # ยกเลิกโหลดโมเดล craft
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()
    exit()

# ตรวจจับข้อความด้วย easyocr
reader = easyocr.Reader(['en'], gpu=False)
results = []

for box in boxes:
    try:
        x_min = int(min(box[:, 0]))
        y_min = int(min(box[:, 1]))
        x_max = int(max(box[:, 0]))
        y_max = int(max(box[:, 1]))

        # ตัดเฉพาะส่วนของภาพ
        cropped = image[y_min:y_max, x_min:x_max]

        # ทำ grayscale และ threshold
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # อ่านข้อความ
        text = reader.readtext(thresh, detail=0)
        joined_text = ' '.join(text).strip()

        results.append((joined_text, (x_min, y_min, x_max, y_max)))
    except Exception as e:
        print("OCR ผิดพลาดที่กล่องหนึ่ง:", e)
        continue

# ยกเลิกการโหลดโมเดล craft
craft.unload_craftnet_model()
craft.unload_refinenet_model()

# เรียงกล่องจากบนลงล่าง ซ้ายไปขวา
results.sort(key=lambda r: (r[1][1], r[1][0]))

# แสดงผล
for text, (x1, y1, x2, y2) in results:
    print(f"ข้อความ: {text} ที่ตำแหน่ง {x1,y1,x2,y2}")

# แสดงภาพพร้อมกล่อง
for text, (x1, y1, x2, y2) in results:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

cv2.imshow("ผลลัพธ์ CRAFT + OCR", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
