import os
import cv2
import numpy as np
from glob import glob

# --- CONFIGURATION ---
dataset_root = "data/training/"
images_dir = os.path.join(dataset_root, "images")
labels_dir = os.path.join(dataset_root, "labels")  # Original TIF masks
output_labels_dir = os.path.join(dataset_root, "labels_yolo")

# Map your folder names to Class IDs
class_map = {
    "beech": 0,
    "larch": 1,
    "oak": 2,
    "birch": 3,
    "magnolia": 4,
}


def convert_mask_to_yolo(mask_path, class_id: int, img_width: int, img_height: int) -> list:
    # Read mask unchanged (keeps bit depth / channels)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return []

    # Convert to single-channel grayscale if needed
    if mask.ndim == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask

    # Ensure 8-bit (findContours requires CV_8UC1)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(src=gray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    # Binarize: treat any non-zero as object
    _, bw = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Quick empty check
    if cv2.countNonZero(bw) == 0:
        return []

    # Find contours on the binary single-channel image
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 50:  # Filter noise
            continue

        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        points = approx.reshape(-1, 2)

        normalized_points = []
        for x, y in points:
            norm_x = max(0.0, min(1.0, float(x) / img_width))
            norm_y = max(0.0, min(1.0, float(y) / img_height))
            normalized_points.extend([norm_x, norm_y])

        line = f"{class_id} " + " ".join(map(str, normalized_points))
        yolo_lines.append(line)

    return yolo_lines


def main():
    os.makedirs(output_labels_dir, exist_ok=True)

    # Look for TIF/TIFF files
    image_extensions = ['*.tif', '*.tiff']
    image_files = []
    for ext in image_extensions:
        # Search recursively in case images are nested, or just in the folder
        image_files.extend(glob(os.path.join(images_dir, ext)))

    print(f"Found {len(image_files)} TIF images. Starting conversion...")

    for img_path in image_files:
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        # print(f'name_no_ext: {name_no_ext}')

        # Read image using UNCHANGED to safely get dimensions
        # (even if 16-bit or multispectral)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # print(f'\timg shape: {img.shape}')

        if img is None:
            print(f"\tCould not read image: {filename}")
            continue

        # Handle case where image has shape (H, W) (grayscale) or (H, W, C)
        if len(img.shape) == 2:
            h, w = img.shape
        else:
            h, w = img.shape[:2]

        all_yolo_lines = []

        # Check every class folder for a matching TIF mask
        for class_name, class_id in class_map.items():
            # Assuming mask filename matches image filename exactly (including .tif extension)
            mask_path = os.path.join(labels_dir, class_name, filename)
            # print(f'\t\tChecking mask path: {mask_path}')

            if os.path.exists(mask_path):
                # print(f'\t\tmask_path "{mask_path}" found for class "{class_name}" (ID: {class_id})')
                lines = convert_mask_to_yolo(mask_path, class_id, w, h)
                # print(f'\t\tConverted {len(lines)} polygons for class "{class_name}"')
                # print(f'\t\tlines: {lines}"')
                all_yolo_lines.extend(lines)

        if all_yolo_lines:
            # Save as .txt (YOLO expects .txt labels even for .tif images)
            txt_filename = name_no_ext + ".txt"
            output_path = os.path.join(output_labels_dir, txt_filename)

            with open(output_path, "w") as f:
                f.write("\n".join(all_yolo_lines))

    print(f"Done! Labels saved to {output_labels_dir}")


if __name__ == "__main__":
    main()
