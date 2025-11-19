from ultralytics import YOLO


def main():
    # 1. Load a pretrained YOLOv11 instance segmentation model
    # model = YOLO('yolo11l-seg.pt')
    # model = YOLO('yolo11n-seg.pt')
    model = YOLO('yolo11s-seg.pt')

    # 2. Start fine-tuning
    print("Starting model training...")
    results = model.train(
        data='tree_dataset_config.yaml',  # Path to your dataset YAML file
        epochs=100,  # Number of epochs to train for
        imgsz=1024,  # Image size (e.g., 640x640)
        batch=8,  # Batch size (adjust based on your GPU VRAM)
        name='yolo11_tree_seg'  # Name for the run
    )
    print("Training completed.")
    print(results)

    # 3. (Optional) Evaluate the model on the validation set
    print("Validating model...")
    metrics = model.val()
    print(metrics)


if __name__ == '__main__':
    main()
