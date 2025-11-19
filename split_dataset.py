import os
import shutil
import random
from glob import glob

# --- CONFIGURATION ---
# Paths to your CURRENT converted data
source_images = "data/training/images_jpg/"  # Folder with your new .jpg files
source_labels = "data/training/labels_yolo/"  # Folder with your .txt files

# Output directory (where the final dataset will be created)
output_root = "data/final_tree_dataset/"

# Split ratio (0.8 = 80% training, 20% validation)
train_ratio = 0.8


# ---------------------

def create_dirs(base_path):
    """Creates the necessary YOLO directory structure."""
    for split in ['train', 'val']:
        os.makedirs(os.path.join(base_path, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'labels', split), exist_ok=True)


def main():
    # 1. Gather all JPG files
    jpg_files = glob(os.path.join(source_images, "*.jpg"))

    # Shuffle to ensure random distribution of tree species
    random.seed(42)  # Seed for reproducibility
    random.shuffle(jpg_files)

    # Calculate split index
    split_index = int(len(jpg_files) * train_ratio)
    train_files = jpg_files[:split_index]
    val_files = jpg_files[split_index:]

    print(f"Total images: {len(jpg_files)}")
    print(f"Training: {len(train_files)} | Validation: {len(val_files)}")

    # 2. Create folder structure
    create_dirs(output_root)

    # 3. Move files
    def move_files(file_list, split_name):
        print(f"Processing {split_name} set...")
        for image_path in file_list:
            filename = os.path.basename(image_path)
            name_no_ext = os.path.splitext(filename)[0]
            label_filename = name_no_ext + ".txt"
            label_path = os.path.join(source_labels, label_filename)

            # Destination paths
            dst_img = os.path.join(output_root, 'images', split_name, filename)
            dst_lbl = os.path.join(output_root, 'labels', split_name, label_filename)

            # Copy Image
            shutil.copy2(image_path, dst_img)

            # Copy Label (only if it exists)
            # If it doesn't exist, it's treated as a background image (no trees)
            if os.path.exists(label_path):
                shutil.copy2(label_path, dst_lbl)

    move_files(train_files, 'train')
    move_files(val_files, 'val')

    print(f"Success! Dataset ready at: {os.path.abspath(output_root)}")


if __name__ == "__main__":
    main()
