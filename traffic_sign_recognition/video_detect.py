import cv2
from ultralytics import YOLO

# Load your trained traffic sign model
model = YOLO("runs/detect/traffic_sign/weights/best.pt")

# Open uploaded video
video_path = "traffic_video.mp4"   # your uploaded video
cap = cv2.VideoCapture(video_path)

# Output video writer
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output_detection.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection (only traffic signs since model is trained for them)
    results = model.predict(frame, conf=0.3, imgsz=640, verbose=False)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Show in window
    cv2.imshow("Traffic Sign Detection - Video", annotated_frame)

    # Save to file
    out.write(cv2.resize(annotated_frame, (width, height)))

    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
