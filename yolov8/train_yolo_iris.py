from ultralytics import YOLO
import torch
import gc


if __name__ == "__main__":

    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)

    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print("GPU name:", device_name)
        print("CUDA version (PyTorch):", torch.version.cuda)

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        torch.cuda.empty_cache()

        print("Using GPU:", device_name)
    else:
        print("Using CPU")
    
    model_name = "yolov8s.pt" # yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

    print(f"\n##### Loading {model_name} model #####\n\n")
    model = YOLO(model_name)

    model.train(
            data='iris.yaml',
            epochs=300, 
            imgsz=640,
            batch=32,
            name='yolov8s_300epochs',
            device=0,
            auto_augment='randaugment'
            )
