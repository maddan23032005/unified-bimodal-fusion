import os
import random
import shutil


base_path = r"C:\Users\madda\OneDrive\Desktop\traffic_sign_recognition\YOLO FORMAT"
images_dir = os.path.join(base_path, "images")
labels_dir = os.path.join(base_path, "labels")


train_images = os.path.join(base_path, "train", "images")
train_labels = os.path.join(base_path, "train", "labels")
val_images = os.path.join(base_path, "val", "images")
val_labels = os.path.join(base_path, "val", "labels")


for path in [train_images, train_labels, val_images, val_labels]:
    os.makedirs(path, exist_ok=True)


image_exts = ['.jpg', '.jpeg', '.png']
image_files = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in image_exts]


random.seed(42)
random.shuffle(image_files)
split_index = int(0.8 * len(image_files))
train_files = image_files[:split_index]
val_files = image_files[split_index:]

for img in train_files:
    img_src = os.path.join(images_dir, img)
    lbl_src = os.path.join(labels_dir, os.path.splitext(img)[0] + ".txt")
    shutil.copy(img_src, os.path.join(train_images, img))
    if os.path.exists(lbl_src):
        shutil.copy(lbl_src, os.path.join(train_labels, os.path.basename(lbl_src)))


for img in val_files:
    img_src = os.path.join(images_dir, img)
    lbl_src = os.path.join(labels_dir, os.path.splitext(img)[0] + ".txt")
    shutil.copy(img_src, os.path.join(val_images, img))
    if os.path.exists(lbl_src):
        shutil.copy(lbl_src, os.path.join(val_labels, os.path.basename(lbl_src)))

print("✅ Dataset split complete:")
print("  Training images:", len(train_files))
print("  Validation images:", len(val_files))
