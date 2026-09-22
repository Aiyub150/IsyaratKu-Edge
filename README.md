# IsyaratKu-edge: Real-Time Hand Gesture Recognition & Translation System

Sistem Pengenalan dan Penerjemahan Bahasa Isyarat (Hand Gesture) secara Real-Time berbasis Machine Learning untuk Lingkungan IoT (**ESP32-CAM + Raspberry Pi 5 / PC**).

---

## 📁 Struktur Folder Proyek

```text
IsyaratKu-edge/
├── configs/
│   ├── labels.json               # 14 kelas gestur (7 Subjek + 7 Predikat)
│   └── settings.py               # Konfigurasi resolusi, buffer, dan path model
│
├── core/                         # LOGIKA ML INTI (Shared antara PC & SBC)
│   ├── __init__.py
│   ├── yolo_detector.py          # Deteksi tangan (YOLOv8n)
│   ├── mediapipe_extractor.py    # Ekstraksi 21 koordinat landmark 3D
│   ├── landmark_normalizer.py    # Normalisasi translasi & skala (wrist-centered)
│   ├── spatial_cnn.py            # 1D-CNN spatial encoder (64-d vector)
│   ├── temporal_lstm.py          # LSTM temporal sequence classifier (12 frames)
│   ├── sentence_builder.py       # State Machine (Subjek + Predikat)
│   └── tts_engine.py             # Non-blocking Text-to-Speech engine
│
├── pc_workspace/                 # LINGKUNGAN PC / LAPTOP (Training & Debugging)
│   ├── training/                 # Skrip training mandiri (ekstraksi landmark, 1D-CNN, LSTM)
│   │   └── 01_extract_landmarks.py
│   ├── testing/                  # MODUL PENGUJIAN DENGAN PERFORMANCE HUD
│   │   ├── test_stream_fps.py      # Uji FPS & latency kamera
│   │   ├── test_yolo_latency.py    # Uji waktu inferensi YOLO (ms)
│   │   ├── test_mediapipe_fps.py   # Uji tracking 21 landmark MediaPipe
│   │   └── test_pipeline_debug.py  # UJI FULL PIPELINE DENGAN VISUAL HUD LENGKAP
│   └── main_pc.py                # Main runner PC (toggle Clean vs Debug mode)
│
├── sbc_workspace/                # LINGKUNGAN SBC (Raspberry Pi 5 - Clean Edge)
│   ├── main_sbc.py               # RUNNER PRODUKSI BERSIH (Zero Rendering Overhead)
│   └── service_health.py         # Systemd health & monitoring reporter (CPU/RAM/Temp)
│
├── data/
│   ├── dataset_raw/              # Video/gambar mentah per kelas (.gitignore)
│   ├── landmarks_cache/          # Dataset koordinat .npy (.gitignore)
│   └── models/                   # Model tersimpan (.onnx, .tflite)
│
├── Feedback/                     # Catatan internal user (Diabaikan oleh Git via .gitignore)
├── requirements-pc.txt           # Dependensi lengkap PC
├── requirements-sbc.txt          # Dependensi ringan SBC
├── .gitignore                    # Konfigurasi pengabaian Git
└── README.md
```

---

## 🖥️ Perbedaan Preview: Testing (PC) vs Implementasi (SBC)

| Fitur | Mode Testing (`pc_workspace/testing/`) | Mode Produksi (`sbc_workspace/main_sbc.py`) |
|---|---|---|
| **Tampilan Visual** | **Rich Performance HUD**: Bounding box YOLO, 21 titik skeleton MediaPipe, garis koneksi sendi berwarna. | **Clean Output**: Hanya menampilkan frame video bersih dengan bar subtitle kalimat terjemahan. |
| **Metrik Performa** | **Tampil Lengkap**: FPS kamera, FPS inferensi, breakdown latensi (ms), confidence score %, status buffer. | **Tidak Ditampilkan**: Menghilangkan seluruh teks performa untuk menghemat 100% beban rendering GPU/CPU di SBC. |
| **Tujuan** | Analisis performa, pencatatan data skripsi/TA, dan debugging model. | Operasional harian di lapangan untuk pengguna masyarakat. |

---

## 🚀 Cara Menjalankan

### 1. Di PC / Laptop (Mode Pengujian & Debug)

```bash
# Install dependensi PC
pip install -r requirements-pc.txt

# Menjalankan Uji Full Pipeline dengan Heads-Up Display (HUD) Lengkap
python pc_workspace/testing/test_pipeline_debug.py

# Menjalankan Uji FPS Kamera
python pc_workspace/testing/test_stream_fps.py

# Menjalankan Uji Latensi YOLO
python pc_workspace/testing/test_yolo_latency.py

# Menjalankan Uji Ekstraksi MediaPipe
python pc_workspace/testing/test_mediapipe_fps.py

# Menjalankan Runner Utama PC (Tekan 'd' untuk toggle debug HUD, 'r' untuk reset kalimat)
python pc_workspace/main_pc.py --debug
```

### 2. Di SBC (Raspberry Pi 5 - Mode Produksi Bersih)

```bash
# Install dependensi ringan SBC
pip install -r requirements-sbc.txt

# Menjalankan runner produksi (Clean View / Subtitle Bar saja)
python sbc_workspace/main_sbc.py

# Menjalankan mode headless (untuk background service systemd)
python sbc_workspace/main_sbc.py --headless

# Cek kondisi kesehatan hardware SBC (CPU, RAM, Suhu)
python sbc_workspace/service_health.py
```

---


