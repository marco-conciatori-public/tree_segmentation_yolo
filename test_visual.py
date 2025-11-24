import os
import cv2
from ultralytics import YOLO
from glob import glob

# --- CONFIGURATION ---
# Path to your trained model
model_path = 'runs/segment/yolo11_tree_seg/weights/best.pt'

# Path to images you want to test (can be a new folder or your val set)
source_images = 'data/final_tree_dataset/images/val/'

# Where to save the results
output_dir = 'output/inference_results/'

# Confidence threshold (0.0 to 1.0)
# Only show detections if model is > 50% confident
conf_threshold = 0.5


# ---------------------

def main():
    # 1. Load the model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # 2. Prepare output folder
    os.makedirs(output_dir, exist_ok=True)

    # 3. Get images
    # Supports jpg, png, tif, etc.
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif']
    image_files = []
    for ext in extensions:
        image_files.extend(glob(os.path.join(source_images, ext)))

    print(f"Found {len(image_files)} images. Running inference...")

    for img_path in image_files:
        filename = os.path.basename(img_path)

        # 4. Run Inference
        # stream=True is efficient for large datasets
        results = model.predict(
            source=img_path,
            conf=conf_threshold,
            save=False,  # We will save manually to control the path
            stream=True,  # Use generator for memory efficiency
            boxes=False,  # Set to False if you only want the mask, no box
            retina_masks=True  # High-res masks (slower but prettier)
        )

        for r in results:
            # --- A. visualization ---
            # Plot the results on the image (returns a BGR numpy array)
            im_array = r.plot()

            # --- B. Print text analysis ---
            # r.boxes.cls contains the class IDs found
            if r.boxes:
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                # Map IDs to names
                species_found = [model.names[i] for i in class_ids]

                # Count occurrences
                from collections import Counter
                counts = Counter(species_found)
                summary = ", ".join([f"{count} {name}" for name, count in counts.items()])
                print(f"[{filename}] Found: {summary}")
            else:
                print(f"[{filename}] No trees detected.")

            # --- C. Save Result ---
            save_path = os.path.join(output_dir, f"res_{filename}.jpg")
            cv2.imwrite(save_path, im_array)

    print(f"Visual results saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
