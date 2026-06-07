# 04_merge_classes.py (Simpler Version)
import os

# We only need to change class 9 into class 8.
CLASS_TO_CHANGE = 9
NEW_CLASS = 8

DATA_DIRS = ["data/train", "data/valid", "data/test"]

def update_labels(directory):
    label_dir = os.path.join(directory, "labels")
    if not os.path.exists(label_dir):
        return 0
    
    count = 0
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(label_dir, filename)
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            updated_lines = []
            made_change = False
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                
                if class_id == CLASS_TO_CHANGE:
                    parts[0] = str(NEW_CLASS)
                    updated_lines.append(" ".join(parts) + "\n")
                    made_change = True
                else:
                    updated_lines.append(line)
            
            if made_change:
                count += 1
                with open(filepath, 'w') as f:
                    f.writelines(updated_lines)
    return count

total_updated = 0
for data_dir in DATA_DIRS:
    num_updated = update_labels(data_dir)
    print(f"Updated {num_updated} label files in '{data_dir}'.")
    total_updated += num_updated
    
print(f"\nTotal label files updated: {total_updated}. Dataset is ready for fine-tuning. ✨")