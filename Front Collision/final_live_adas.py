import cv2
import time
from ultralytics import YOLO
from collections import deque

# --- Configuration ---
MODEL_PATH = 'runs/detect/yolov8n_adas_finetuned3/weights/best.pt'
CONFIDENCE_THRESHOLD = 0.45
COLLISION_CLASSES = {0, 1, 2, 10} # 'biker', 'car', 'pedestrian', 'truck'

# --- Enhanced Warning System Configuration ---
# Thresholds are percentages of the total frame area
WARNING_THRESHOLD = 0.18  # 18% - Triggers a RED collision warning
CAUTION_THRESHOLD = 0.10  # 10% - Triggers a YELLOW caution warning

# History buffer to stabilize warnings (averages the size over the last 5 frames)
TRACKING_HISTORY = 5

def main():
    # Load your custom-trained YOLOv8 model
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("✅ Enhanced Live ADAS feed started. Press 'q' to quit.")

    # Variables for FPS calculation and tracking history
    last_time = time.time()
    tracked_objects = {} # Stores history of tracked objects {id: deque([area1, area2, ...])}

    while True:
        # Read a frame from the webcam
        success, frame = cap.read()
        if not success:
            print("Warning: Failed to grab frame. Skipping...")
            if cv2.waitKey(100) & 0xFF == ord('q'): break
            continue

        frame_height, frame_width, _ = frame.shape
        frame_area = frame_height * frame_width
        
        # This is the highest warning level for the current frame (0=Safe, 1=Caution, 2=Warning)
        max_warning_level = 0 

        # --- Use YOLOv8's Tracking Feature ---
        # persist=True tells the tracker that the new frame is from the same video sequence
        results = model.track(frame, persist=True, verbose=False)

        # Process the results
        if results[0].boxes.id is not None: # Check if tracking IDs are present
            boxes = results[0].boxes.cpu().numpy()
            for box in boxes:
                track_id = int(box.id[0])
                confidence = float(box.conf[0])
                
                if confidence > CONFIDENCE_THRESHOLD:
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Initialize color and warning level for this object
                    color = (0, 255, 0) # Green (Safe)
                    warning_level = 0

                    # --- Stable Warning Logic ---
                    if cls_id in COLLISION_CLASSES:
                        box_area = (x2 - x1) * (y2 - y1)
                        
                        # Add current area to the object's history
                        if track_id not in tracked_objects:
                            tracked_objects[track_id] = deque(maxlen=TRACKING_HISTORY)
                        tracked_objects[track_id].append(box_area)

                        # Calculate average area from history for stability
                        avg_area = sum(tracked_objects[track_id]) / len(tracked_objects[track_id])
                        
                        # Check against thresholds
                        if (avg_area / frame_area) > WARNING_THRESHOLD:
                            warning_level = 2 # Red (Warning)
                            color = (0, 0, 255)
                        elif (avg_area / frame_area) > CAUTION_THRESHOLD:
                            warning_level = 1 # Yellow (Caution)
                            color = (0, 255, 255)

                    # Update the frame's maximum warning level
                    if warning_level > max_warning_level:
                        max_warning_level = warning_level

                    # Draw bounding box and label
                    class_name = model.names[cls_id]
                    label = f'ID:{track_id} {class_name} {confidence:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # --- On-Screen Display (OSD) / Dashboard ---
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - last_time)
        last_time = current_time

        # Create a semi-transparent background for the OSD
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame_width, 60), (0, 0, 0), -1)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (15, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display System Status based on the highest warning level
        if max_warning_level == 2:
            status_text = "COLLISION WARNING!"
            status_color = (0, 0, 255)
        elif max_warning_level == 1:
            status_text = "CAUTION"
            status_color = (0, 255, 255)
        else:
            status_text = "SYSTEM ACTIVE"
            status_color = (0, 255, 0)
        
        cv2.putText(frame, status_text, (frame_width - 350, 40), 
                    cv2.FONT_HERSHEY_TRIPLEX, 1.2, status_color, 2)

        # Show the final frame
        cv2.imshow("Enhanced Live ADAS Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    main()