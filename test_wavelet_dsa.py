import torch
from ultralytics import YOLO

def main():
    try:
        print("Membangun model YOLO11 Wavelet dengan DSA...")
        model = YOLO("ultralytics/cfg/models/11/yolo11-wavelet-thesis.yaml")
        
        print("Model berhasil dibangun! Menguji forward pass dengan data dummy...")
        dummy_input = torch.randn(1, 3, 640, 640)
        
        device = next(model.model.parameters()).device
        dummy_input = dummy_input.to(device)
        
        # Jalankan forward pass langsung ke arsitektur PyTorch
        model.model.eval()
        with torch.no_grad():
            output = model.model(dummy_input)
        
        print("\n✅ Uji coba sukses! Tidak ada error saat inisialisasi maupun forward pass.")
        
        # Coba cek layer WaveletUp untuk membuktikan DSA bekerja
        print("\nMengecek keberadaan bobot Attention dari DirectionalSubbandAttention (DSA):")
        found = False
        for i, m in enumerate(model.model.model):
            if m.__class__.__name__ == "WaveletUp":
                found = True
                attn = m.last_attn
                if attn is not None:
                    print(f"  → Layer {i} (WaveletUp): bobot attention berhasil disimpan dengan shape {attn.shape}")
                    print(f"    Nilai contoh: LH={attn[0, 0].item():.4f}, HL={attn[0, 1].item():.4f}, HH={attn[0, 2].item():.4f}")
                else:
                    print(f"  → Layer {i} (WaveletUp): Tidak ada bobot attention (mungkin use_dsa=False)")
        
        if not found:
            print("Peringatan: Tidak ditemukan layer WaveletUp pada model ini.")
            
    except Exception as e:
        import traceback
        print("\n❌ Error ditemukan:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
