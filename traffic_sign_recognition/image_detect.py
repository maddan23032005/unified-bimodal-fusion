from ultralytics import YOLO
import cv2
from tkinter import Tk, filedialog

model = YOLO("runs/detect/traffic_sign/weights/best.pt")

Tk().withdraw()  
image_path = filedialog.askopenfilename(
    title="Select an image for detection",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
)

if not image_path:
    print("No image selected.")
    exit()

image = cv2.imread(image_path)

results = model(image)

annotated_frame = results[0].plot()
cv2.imshow("Detection Result", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
