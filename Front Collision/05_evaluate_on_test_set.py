# 05_evaluate_on_test_set.py
from ultralytics import YOLO

# --- Configuration ---
# Path to your BEST fine-tuned model
MODEL_PATH = 'runs/detect/yolov8n_adas_finetuned3/weights/best.pt'

# Path to your data configuration file
DATA_YAML_PATH = 'data/data.yaml'

def main():
    # Load your custom-trained YOLOv8 model
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please make sure the model path is correct: '{MODEL_PATH}'")
        return

    print("Starting evaluation on the test set... 🚀")

    # Evaluate the model on the 'test' split defined in the data.yaml file
    metrics = model.val(
        data=DATA_YAML_PATH,
        split='test',
        name='yolov8n_test_evaluation'
    )
    
    print("\n--- Test Set Performance Metrics ---")
    print(f"Overall mAP50-95: {metrics.box.map:.4f}")
    print(f"Overall mAP50: {metrics.box.map50:.4f}")
    print(f"Overall mAP75: {metrics.box.map75:.4f}")
    print("------------------------------------")
    # metrics.box.maps is a per-class mAP50-95 list
    # for i, map_value in enumerate(metrics.box.maps):
    #     print(f"Class '{model.names[i]}': mAP50-95 = {map_value:.4f}")

if __name__ == "__main__":
    main()