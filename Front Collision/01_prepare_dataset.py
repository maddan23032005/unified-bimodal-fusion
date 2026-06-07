# 01_prepare_dataset.py

import os
import random
import shutil

# --- Configuration ---
DATA_SOURCE_DIR = r"C:\Users\madda\OneDrive\Desktop\Front Collision\data\export"
TRAIN_RATIO = 0.70
VALID_RATIO = 0.20
# TEST_RATIO will be the remainder (0.10)

# --- Paths ---
base_dir = "data"
image_source_dir = os.path.join(DATA_SOURCE_DIR, "images")
label_source_dir = os.path.join(DATA_SOURCE_DIR, "labels")

# --- Create destination directories ---
sets = ["train", "valid", "test"]
for s in sets:
    os.makedirs(os.path.join(base_dir, s, "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, s, "labels"), exist_ok=True)

print("Created train, valid, and test directories.")

# --- Get all image filenames (without extension) ---
all_files = [os.path.splitext(f)[0] for f in os.listdir(image_source_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(all_files)

# --- Calculate split indices ---
total_files = len(all_files)
train_end = int(total_files * TRAIN_RATIO)
valid_end = int(total_files * (TRAIN_RATIO + VALID_RATIO))

# --- Assign files to sets ---
train_files = all_files[:train_end]
valid_files = all_files[train_end:valid_end]
test_files = all_files[valid_end:]

datasets = {
    "train": train_files,
    "valid": valid_files,
    "test": test_files
}

# --- Copy files to their respective directories ---
def copy_files(file_list, set_name):
    for filename in file_list:
        # Source paths
        img_src_path = os.path.join(image_source_dir, f"{filename}.jpg") # Assuming .jpg, change if needed
        label_src_path = os.path.join(label_source_dir, f"{filename}.txt")

        # Destination paths
        img_dest_path = os.path.join(base_dir, set_name, "images", f"{filename}.jpg")
        label_dest_path = os.path.join(base_dir, set_name, "labels", f"{filename}.txt")

        # Copy files
        if os.path.exists(img_src_path):
            shutil.copy(img_src_path, img_dest_path)
        if os.path.exists(label_src_path):
            shutil.copy(label_src_path, label_dest_path)

for set_name, file_list in datasets.items():
    copy_files(file_list, set_name)
    print(f"Copied {len(file_list)} files to '{set_name}' set.")

print("\nDataset splitting complete! 🎉")