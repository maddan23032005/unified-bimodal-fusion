import cv2
import torch
from ultralytics import YOLO

# --- settings ---
WEIGHTS = r"runs/detect/traffic_sign/weights/best.pt"
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
CONF_TH = 0.65      # higher conf = fewer false positives
IOU_TH = 0.5
MIN_AREA = 30 * 30  # ignore boxes smaller than 30x30 px

# Load model on GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(WEIGHTS).to(device)

# Allow ALL traffic-sign classes (0..42)
if isinstance(model.names, dict):
    allowed_classes = list(model.names.keys())
else:
    allowed_classes = list(range(len(model.names)))

print(f"Loaded {len(model.names)} classes.")

# Webcam
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Inference (filter to allowed classes)
    results = model.predict(
        source=frame,
        device=device,
        imgsz=512,
        conf=CONF_TH,
        iou=IOU_TH,
        classes=allowed_classes,
        verbose=False
    )

    # Draw detections
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()

            # size filter to remove tiny false positives
            if (x2 - x1) * (y2 - y1) < MIN_AREA:
                continue

            label = f"{model.names[cls_id]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 - Traffic Sign Detection (All 43 classes)", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord(']'):  # increase confidence
        CONF_TH = min(0.95, CONF_TH + 0.05); print("CONF_TH ->", CONF_TH)
    elif k == ord('['):  # decrease confidence
        CONF_TH = max(0.10, CONF_TH - 0.05); print("CONF_TH ->", CONF_TH)

cap.release()
cv2.destroyAllWindows()
