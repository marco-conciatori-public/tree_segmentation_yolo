import torch
from ultralytics import YOLO


def get_available_device(verbose: int = 0) -> torch.device:
    if not torch.cuda.is_available():
        print('GPU not found, using CPU')
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')

    if verbose >= 1:
        print(f'Device: {device}')
    return device


def main():
    # 1. Load a pretrained YOLOv11 instance segmentation model
    # model = YOLO('yolo11l-seg.pt')
    # model = YOLO('yolo11n-seg.pt')
    model = YOLO('yolo11s-seg.pt')
    device = get_available_device(verbose=1)

    # 2. Start fine-tuning
    print("Starting model training...")
    results = model.train(
        data='tree_dataset_config.yaml',  # Path to your dataset YAML file
        epochs=100,  # Number of epochs to train for
        imgsz=1024,  # Image size (e.g., 640x640)
        batch=8,  # Batch size (adjust based on your GPU VRAM)
        name='yolo11_tree_seg',  # Name for the run
        device=device  # Use GPU 0; change if necessary
    )
    print("Training completed.")
    print(results)

    # 3. (Optional) Evaluate the model on the validation set
    print("Validating model...")
    metrics = model.val()
    print(metrics)


if __name__ == '__main__':
    main()
