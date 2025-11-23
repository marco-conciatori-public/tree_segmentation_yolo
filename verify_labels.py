import os
import cv2
import numpy as np
from glob import glob

# --- CONFIGURATION ---
# Path to the dataset you want to check
# We usually check the 'train' folder as it has the most data
dataset_subset = "data/final_tree_dataset/images/train"
labels_subset = "data/final_tree_dataset/labels/train"

# Output folder for the verification images
output_dir = "output/verification_masks"


# ---------------------

def get_coords_from_yolo(line, img_w, img_h):
    """Parses a YOLO line and returns pixel coordinates."""
    parts = line.strip().split()
    # class_id = int(parts[0]) # We don't need class ID for binary mask

    # The rest are coordinates: x1 y1 x2 y2 ...
    coords = [float(x) for x in parts[1:]]

    # Reshape into pairs of (x, y)
    points = []
    for i in range(0, len(coords), 2):
        x_norm = coords[i]
        y_norm = coords[i + 1]

        # Denormalize: multiply by image dimensions
        x_pixel = int(x_norm * img_w)
        y_pixel = int(y_norm * img_h)
        points.append([x_pixel, y_pixel])

    return np.array(points, dtype=np.int32)


def main():
    os.makedirs(output_dir, exist_ok=True)

    # Get images
    img_paths = glob(os.path.join(dataset_subset, "*.jpg"))

    # Limit checking to first 20 images to save time (remove slice to check all)
    img_paths = img_paths[:20]

    print(f"Checking {len(img_paths)} images...")

    for img_path in img_paths:
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        label_path = os.path.join(labels_subset, name_no_ext + ".txt")

        # 1. Load Image (Needed for dimensions)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # 2. Create a Black Canvas
        mask = np.zeros(shape=(h, w), dtype=np.uint8)

        # 3. Parse Label File
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                # Get polygon points
                pts = get_coords_from_yolo(line, w, h)

                # Draw white polygon on black mask
                # cv2.fillPoly expects a list of arrays
                cv2.fillPoly(mask, [pts], 255)

        # 4. Save the B&W Mask
        cv2.imwrite(os.path.join(output_dir, f"{name_no_ext}_mask.png"), mask)

        # --- OPTIONAL: Save an Overlay for easier debugging ---
        # Convert mask to color so we can overlay it (Green color)
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_color[:, :, 0] = 0  # Zero Blue
        mask_color[:, :, 2] = 0  # Zero Red
        # Now mask_color is pure Green where the tree is

        # Blend images: 70% original image + 30% green mask
        overlay = cv2.addWeighted(img, 0.7, mask_color, 0.3, 0)

        # Draw the polygon outline in Red for extra clarity
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    pts = get_coords_from_yolo(line, w, h)
                    cv2.polylines(overlay, [pts], True, (0, 0, 255), 2)

        cv2.imwrite(os.path.join(output_dir, f"{name_no_ext}_overlay.jpg"), overlay)

    print(f"Check complete! Open the folder '{output_dir}' to verify.")


if __name__ == "__main__":
    main()
