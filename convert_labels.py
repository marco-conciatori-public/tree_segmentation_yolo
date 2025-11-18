import os
import cv2
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


# ---------------------

def convert_mask_to_yolo(mask_path, class_id: int, img_width: int, img_height: int) -> list:
    # Read mask as grayscale.
    # Note: If your TIF masks are not 0-255 (e.g. 0-1 binary),
    # OpenCV handles them, but we ensure we treat non-zero as the object.
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path)
    print("mask")
    print(mask.shape)
    # check if mask is all zeros
    if mask is not None and cv2.countNonZero(mask[:, :, 0]) == 0:
        print('mask is all zeros in channel 0')
    if mask is not None and cv2.countNonZero(mask[:, :, 1]) == 0:
        print('mask is all zeros in channel 1')
    if mask is not None and cv2.countNonZero(mask[:, :, 2]) == 0:
        print('mask is all zeros in channel 2')

    if mask is None:
        return []

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("contours")
    print(contours)

    yolo_lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 50:  # Filter noise
            continue

        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        points = approx.flatten().reshape(-1, 2)

        normalized_points = []
        for x, y in points:
            # Normalize and Clamp
            norm_x = max(0, min(1, x / img_width))
            norm_y = max(0, min(1, y / img_height))
            normalized_points.extend([norm_x, norm_y])

        line = f"{class_id} " + " ".join(map(str, normalized_points))
        yolo_lines.append(line)

    print("yolo_lines")
    print(yolo_lines)
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
        print(f'name_no_ext: {name_no_ext}')

        # Read image using UNCHANGED to safely get dimensions
        # (even if 16-bit or multispectral)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # print(f'img shape: {img.shape}')

        if img is None:
            print(f"Could not read image: {filename}")
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
            # print(f'\tChecking mask path: {mask_path}')

            if os.path.exists(mask_path):
                print(f'\t mask_path "{mask_path}" found for class "{class_name}" (ID: {class_id})')
                lines = convert_mask_to_yolo(mask_path, class_id, w, h)
                # print(f'\t Converted {len(lines)} polygons for class "{class_name}"')
                # print(f'\t lines: {lines}"')
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
