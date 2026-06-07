# 02_train_model.py (Updated for Fine-Tuning)
from ultralytics import YOLO

def main():
    # Load your best performing model as the starting point
    model = YOLO('runs/detect/yolov8n_adas_gpu/weights/best.pt')

    # Path to your updated data.yaml file
    data_yaml_path = 'data/data.yaml'

    print("Starting fine-tuning to improve weak classes... 🎯")
    
    results = model.train(
        data=data_yaml_path,
        imgsz=640,
        epochs=75,  # Train for 25 more epochs (50 + 25)
        batch=4,
        copy_paste=0.1,  # Add copy-paste augmentation
        name='yolov8n_adas_finetuned',
        device=0
    )

    print("Fine-tuning finished! ✅")

if __name__ == '__main__':
    main()