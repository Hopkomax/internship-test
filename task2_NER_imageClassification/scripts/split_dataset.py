import os
import shutil
import random

# Paths
DATA_DIR = "data/images"  # Original dataset
TRAIN_DIR = "data/train"  # Training set
VAL_DIR = "data/val"      # Validation set
SPLIT_RATIO = 0.8         # 80% Train, 20% Validation

# Ensure train and validation directories exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# Loop through each animal class
for class_name in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_name)

    if os.path.isdir(class_path):  # Only process folders
        images = os.listdir(class_path)
        random.shuffle(images)  # Shuffle images to avoid ordering bias

        # Split images into train/val
        split_index = int(len(images) * SPLIT_RATIO)
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Create class folders in train/val
        os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, class_name), exist_ok=True)

        # Move images to train folder
        for img in train_images:
            src_path = os.path.join(class_path, img)
            dest_path = os.path.join(TRAIN_DIR, class_name, img)
            shutil.copy(src_path, dest_path)

        # Move images to validation folder
        for img in val_images:
            src_path = os.path.join(class_path, img)
            dest_path = os.path.join(VAL_DIR, class_name, img)
            shutil.copy(src_path, dest_path)

        print(f"✅ {class_name}: {len(train_images)} train, {len(val_images)} val")

print("\n🎯 Dataset successfully split into Train/Validation sets!")
