# detect_on_video.py
import cv2
import argparse
from ultralytics import YOLO

# --- Configuration ---
# Path to your best model
MODEL_PATH = 'runs/detect/yolov8n_adas_finetuned3/weights/best.pt'
CONFIDENCE_THRESHOLD = 0.4

# We use the original collision classes, including pedestrians, for video files.
COLLISION_CLASSES = {0, 1, 2, 10} # 'biker', 'car', 'pedestrian', 'truck'

# Use the more sensitive threshold for real driving footage.
AREA_THRESHOLD = 0.15 

def main(input_path, output_path):
    # Load your model
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize video capture from the input file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at '{input_path}'")
        return

    # Get video properties to create the output file
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer to save the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print("✅ Processing video... Press 'q' to quit early.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Finished processing video.")
            break

        frame_area = frame_width * frame_height
        collision_warning = False

        # Perform object detection
        results = model(frame, stream=True, verbose=False)

        # Process results
        for r in results:
            for box in r.boxes:
                confidence = float(box.conf[0])
                
                if confidence > CONFIDENCE_THRESHOLD:
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_name = model.names[cls_id]

                    color = (0, 255, 0) # Green
                    if cls_id in COLLISION_CLASSES:
                        box_area = (x2 - x1) * (y2 - y1)
                        if (box_area / frame_area) > AREA_THRESHOLD:
                            collision_warning = True
                            color = (0, 0, 255) # Red

                    label = f'{class_name} {confidence:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if collision_warning:
            cv2.putText(frame, "COLLISION WARNING!", (50, 50), 
                        cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 3)

        # Write the processed frame to the output video file
        out.write(frame)

        # Display the frame in a window
        cv2.imshow("Video Processing", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processing stopped by user.")
            break

    # Release all resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Output video saved to '{output_path}'")

if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Process a video file for object detection.")
    parser.add_argument("--input", required=True, help="Path to the input video file.")
    parser.add_argument("--output", default="output_video.mp4", help="Path to save the output video file.")
    args = parser.parse_args()
    
    main(args.input, args.output)