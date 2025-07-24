import os
import random
import shutil

# Set seed for reproducibility
random.seed(42)

# Define paths
image_train_dir = 'training/data/images/train'
image_val_dir = 'training/data/images/val'
label_train_dir = 'training/data/labels/train'
label_val_dir = 'training/data/labels/val'

# Create val directories if they don't exist
os.makedirs(image_val_dir, exist_ok=True)
os.makedirs(label_val_dir, exist_ok=True)

# Get all image files from training directory
image_files = [f for f in os.listdir(image_train_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Calculate number of images to move (10%)
num_to_move = int(0.10 * len(image_files))
val_files = random.sample(image_files, num_to_move)

for filename in val_files:
    base_name = os.path.splitext(filename)[0]
    label_file = base_name + '.txt'

    # Move image
    src_img_path = os.path.join(image_train_dir, filename)
    dst_img_path = os.path.join(image_val_dir, filename)
    shutil.move(src_img_path, dst_img_path)

    # Move corresponding label file
    src_label_path = os.path.join(label_train_dir, label_file)
    dst_label_path = os.path.join(label_val_dir, label_file)

    if os.path.exists(src_label_path):
        shutil.move(src_label_path, dst_label_path)
    else:
        print(f"[Warning] No label found for: {filename}")

print(f"\n✅ Moved {num_to_move} images (and their labels) to validation set.")
