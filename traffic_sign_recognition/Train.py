from ultralytics import YOLO

def main():
    # Load the pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # Train the model
    model.train(
        task="detect",
        mode="train",
        data=r"C:\Users\madda\OneDrive\Desktop\traffic_sign_recognition\YOLO FORMAT\data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        name="traffic_sign"
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Important for Windows multiprocessing support
    main()
