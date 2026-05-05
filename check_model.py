import torch
from ultralytics import YOLO

def test_custom_model():
    print("Mencoba memuat model kustom...")
    try:
        # 1. Inisialisasi model dari YAML
        model = YOLO("ultralytics/cfg/models/11/yolo11-wavelet-thesis.yaml")
        
        # 2. Cetak ringkasan arsitektur
        model.info()
        print("\nArsitektur berhasil dimuat!")

        # 3. Uji Forward Pass dengan dummy data (Batch size=1, Channel=3, 640x640)
        dummy_input = torch.randn(1, 3, 640, 640)
        print("Menjalankan Forward Pass...")
        results = model.predict(dummy_input, verbose=False)
        
        print(f"Berhasil! Output shape: {results[0].boxes.shape if results[0].boxes is not None else 'No detections'}")
        print("Model siap digunakan untuk training.")

    except Exception as e:
        print(f"\nERROR terdeteksi: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_custom_model()