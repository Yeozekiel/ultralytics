from ultralytics import YOLO

def main():
    # Load the custom wavelet model configuration
    print("Loading YOLO11 Wavelet model...")
    model = YOLO("ultralytics/cfg/models/11/yolo11-wavelet-parallel.yaml") # Pendekatan Paralel Ekstraktor Tepi
    # model = YOLO("ultralytics/cfg/models/11/yolo11-wavelet-dsa.yaml")
    # model = YOLO("ultralytics/cfg/models/11/yolo11-wavelet-nodsa.yaml") # Uncomment ini untuk Ablation Study
    # model = YOLO("ultralytics/cfg/models/11/yolo11n.yaml")
    
    # Start the training process
    # Note: workers=0 is recommended on Windows to avoid dataloader multiprocessing deadlocks
    print("Starting training process...")
    results = model.train(
        data="dataset_vindr.yaml",
        epochs=200,            # Adjust epochs as necessary
        imgsz=640,             # Target image size
        batch=16,              # Adjust based on your GPU VRAM
        device=0,              # Use GPU 0
        workers=0,             # Prevents multiprocessing errors on Windows
        amp=False,             # Disable mixed precision to avoid pytorch_wavelets fp16 errors
        mosaic=0.0,
        mixup=0.0,
        project="runs/train",
        name="yolo11_wavelet_thesis_vindr_hybrid_activated_no_mosaic_mixup_nodsa"
        # name="yolo11n_vindr_no_mosaic_mixup"
    )

if __name__ == "__main__":
    main()
