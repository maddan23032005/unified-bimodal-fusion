import os
from PIL import Image
import shutil

# Paths
input_dir ="C:\Users\madda\OneDrive\Desktop\traffic_sign_recognition\Traffic Sign Dataset\Train"
output_img_dir = "C:\Users\madda\OneDrive\Desktop\traffic_sign_recognition\YOLO FORMAT\IMAGES"
output_lbl_dir = "C:\Users\madda\OneDrive\Desktop\traffic_sign_recognition\YOLO FORMAT\LABELS"

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

# Loop through each class folder (0 to 42)
for class_id in range(43):
    class_folder = os.path.join(input_dir, str(class_id))  # ✅ FIXED: no zero-padding
    if not os.path.exists(class_folder):
        print(f"⚠️ Warning: {class_folder} not found.")
        continue

    for img_file in os.listdir(class_folder):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm')):  # safer lowercase check
            src_path = os.path.join(class_folder, img_file)
            
            # Rename image to avoid duplicates
            new_img_name = f"{class_id}_{img_file}"
            dst_img_path = os.path.join(output_img_dir, new_img_name)
            shutil.copy(src_path, dst_img_path)

            # Generate dummy YOLO label (full bounding box, class_id is folder name)
            label_path = os.path.join(output_lbl_dir, os.path.splitext(new_img_name)[0] + '.txt')
            with open(label_path, 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")  # center_x, center_y, width, height (normalized)

print("✅ Conversion complete. YOLO-format dataset ready.")
