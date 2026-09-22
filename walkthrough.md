# Walkthrough: Pembangunan Struktur ML IsyaratKu-edge (PC & SBC)

Pembangunan struktur folder, modul Machine Learning inti, modul pengujian dengan **Performance HUD**, dan runner produksi **Clean Output** untuk proyek **`IsyaratKu-edge`** telah berhasil diselesaikan.

---

## 1. Komponen yang Telah Dibuat

### A. Konfigurasi & Keamanan Git
- **`.gitignore`**: Dikonfigurasi agar folder `Feedback/`, cache dataset (`data/dataset_raw/`, `data/landmarks_cache/`), dan file database (`*.db`, `*.sqlite3`) tidak terunggah ke repositori GitHub.
- **`Feedback/`**: Folder privat untuk catatan internal dan masukan pengguna (diabaikan oleh Git).
- **`configs/labels.json`**: Memuat 14 kelas gestur terstruktur (7 Subjek: *saya, kamu, anda, kami, kita, dia, mereka*; 7 Predikat: *makan, minum, tidur, belajar, bekerja, berjalan, membaca*).
- **`configs/settings.py`**: Parameter global resolusi (VGA 640x480), sequence length (12 frame), threshold confidence (0.65), dan debounce.

### B. Core Machine Learning Modules (`core/`)
- **`core/landmark_normalizer.py`**: Normalisasi 21 koordinat landmark 3D (wrist-centered + scale invariant) menghasilkan vektor 63 float.
- **`core/sentence_builder.py`**: State machine perangkai kalimat (Subjek + Predikat) dengan filter debounce dan cooldown.
- **`core/mediapipe_extractor.py`**: Wrapper MediaPipe Hands dengan mode Clean (SBC) dan mode Debug (PC).
- **`core/yolo_detector.py`**: Wrapper deteksi keberadaan tangan YOLOv8n (mendukung ONNX dan PyTorch).
- **`core/spatial_cnn.py`**: Encoder spasial 1D-CNN (vektor 63-d ➔ 64-d).
- **`core/temporal_lstm.py`**: Classifier sekuens temporal LSTM dengan FIFO buffer 12 frame.
- **`core/tts_engine.py`**: Engine Text-to-Speech non-blocking thread offline (`pyttsx3`).

### C. Modul Testing dengan Performance HUD (`pc_workspace/testing/`)
- **`test_pipeline_debug.py`**: Modul uji full pipeline dengan **Heads-Up Display (HUD)**:
  - Indikator FPS kamera & FPS inferensi
  - Breakdown latensi per tahap: YOLO (ms), MediaPipe (ms), Model (ms), Total (ms)
  - Overlay 21 titik skeleton tangan berwarna & bounding box
  - Status state machine kalimat & confidence bar
- **`test_stream_fps.py`**: Uji FPS dan latensi streaming kamera/ESP32-CAM.
- **`test_yolo_latency.py`**: Uji akurasi dan waktu inferensi detektor YOLO.
- **`test_mediapipe_fps.py`**: Uji tracking dan deteksi 21 landmark MediaPipe.

### D. SBC Production Runtime (`sbc_workspace/`)
- **`main_sbc.py`**: Runner produksi bersih untuk **Raspberry Pi 5**:
  - **Zero Rendering Overhead**: Menghilangkan seluruh teks FPS, latensi, dan gambar skeleton.
  - Menampilkan video bersih dengan bar subtitle kalimat terjemahan yang elegan (atau mode `--headless`).
  - Output audio TTS langsung aktif saat kalimat lengkap.
- **`service_health.py`**: Pemantauan kesehatan hardware SBC (CPU, RAM, Suhu °C) untuk integrasi `systemd`.

---

## 2. Cara Menjalankan Modul

### Mode Testing (Dengan Metrik Performa)
```bash
python pc_workspace/testing/test_pipeline_debug.py
```

### Mode Produksi SBC (Clean Output Tanpa Metrik)
```bash
python sbc_workspace/main_sbc.py
```
