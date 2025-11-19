import os
import cv2
import numpy as np
from glob import glob

# --- CONFIGURATION ---
dataset_root = "path/to/your/dataset_root"
input_tif_dir = os.path.join(dataset_root, "images")  # Where your TIFs are now
output_jpg_dir = os.path.join(dataset_root, "images_jpg")  # Where to save new JPGs


# ---------------------

def main():
    # Create output directory if it doesn't exist
    os.makedirs(output_jpg_dir, exist_ok=True)

    # Find all TIF images
    tif_files = glob(os.path.join(input_tif_dir, "*.tif")) + \
                glob(os.path.join(input_tif_dir, "*.tiff"))

    print(f"Found {len(tif_files)} TIF files. Converting to JPG...")

    for tif_path in tif_files:
        filename = os.path.basename(tif_path)
        name_no_ext = os.path.splitext(filename)[0]

        # Read image unchanged (to detect 16-bit or Alpha channels)
        img = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"Error reading: {filename}")
            continue

        # --- STEP 1: Handle 16-bit images ---
        # YOLO expects 8-bit (0-255). If input is 16-bit (0-65535), we must scale it.
        if img.dtype == np.uint16:
            # Normalize to 0-255 range
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype(np.uint8)

        # --- STEP 2: Handle Channel Counts ---
        # Case: Grayscale (2D array) -> Convert to BGR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Case: 4-Channel (RGBA) -> Drop Alpha channel
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Case: More than 4 channels (Multispectral) -> Keep first 3 (usually RGB)
        elif len(img.shape) == 3 and img.shape[2] > 4:
            img = img[:, :, :3]

        # --- STEP 3: Save as JPG ---
        # We use the same 'name_no_ext' so it matches your .txt labels
        output_path = os.path.join(output_jpg_dir, name_no_ext + ".jpg")

        # quality=95 is a good balance (default is usually 95)
        cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    print(f"Done! Converted images saved to: {output_jpg_dir}")


if __name__ == "__main__":
    main()
