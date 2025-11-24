from ultralytics import YOLO

# --- CONFIGURATION ---
model_path = 'runs/segment/yolo11_tree_seg/weights/best.pt'
data_yaml = 'tree_dataset_config.yaml'  # Ensure this points to your yaml file


# ---------------------

def main():
    # Load the model
    model = YOLO(model_path)

    print("Starting model test...")

    # Run validation
    # split='val' uses the validation set.
    # If you have a separate 'test' set in your YAML, change split to 'test'
    metrics = model.val(
        data=data_yaml,
        split='test',
        imgsz=1024,
        batch=8,
        conf=0.001,  # Lower conf is standard for calculating mAP
        iou=0.6  # NMS IoU threshold
    )

    # Print specific segmentation metrics
    print("\n" + "=" * 30)
    print("RESULTS SUMMARY")
    print("=" * 30)

    # map50-95 is the most robust metric (average mAP over different IoU thresholds)
    print(f"mAP(50-95): {metrics.seg.map:.4f}")
    print(f"mAP(50):    {metrics.seg.map50:.4f}")

    # You can also access per-class metrics if needed
    print("\nPer-class mAP(50-95):")
    for i, name in enumerate(model.names.values()):
        # Note: metrics.seg.maps is an array of mAP50-95 per class
        print(f"  {name}: {metrics.seg.maps[i]:.4f}")


if __name__ == "__main__":
    main()
