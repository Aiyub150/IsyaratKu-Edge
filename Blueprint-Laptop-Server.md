# Roadmap Perancangan dan Pembangunan Machine Learning untuk Deteksi Hand Gesture

> Dokumen kerja ini merupakan **blueprint teknis pengembangan** untuk tahap pembangunan sistem setelah proposal tugas akhir ditetapkan. Isi dokumen mempertahankan komponen yang sudah ada pada proposal dan menambahkan beberapa rancangan teknis untuk membantu proses implementasi.
>
> **Judul acuan:** *Implementasi Machine Learning untuk Deteksi Hand Gesture pada Sistem IoT Berbasis ESP32-CAM*
>
> **Prinsip utama:** tambahan teknis di dokumen ini **tidak menggantikan** isi proposal. Fitur tambahan ditempatkan sebagai pengembangan teknis, eksperimen, atau tahap lanjutan yang tetap berada di sekitar ruang lingkup sistem.

---

## 1. Dasar Perancangan

Sistem yang dibangun tetap berpusat pada ESP32-CAM sebagai perangkat akuisisi citra. ESP32-CAM menangkap citra tangan secara real-time dan mengirimkan data melalui Wi-Fi menuju sistem pemrosesan. Server/laptop menjalankan komponen machine learning dan mengembalikan hasil deteksi kepada aplikasi. Rancangan ini sesuai dengan bagian desain sistem pada proposal.

Komponen machine learning yang dipertahankan dan dioptimalkan adalah:

- YOLO (YOLOv8n) untuk deteksi cepat dan lokalisasi bounding box area tangan.
- MediaPipe Hands untuk mengekstraksi 21 koordinat landmark 3D tangan.
- 1D-CNN untuk pembelajaran fitur spasial dari representasi vektor 21 landmark tangan (63 nilai float).
- LSTM untuk pemrosesan urutan data temporal antar frame.
- Sentence Builder/temporary buffer untuk membentuk kalimat sederhana.
- Text-to-Speech (TTS) untuk menghasilkan keluaran suara.

Lapisan aplikasi mencakup:

- Website sebagai antarmuka sistem (FastAPI + Jinja2/Web UI).
- SQLite sebagai penyimpanan lokal yang efisien, tanpa daemon, berkinerja tinggi, dan mendukung tipe data JSON untuk data pengguna, konfigurasi model, riwayat deteksi, dan log sistem.
- Dua hak akses: **Developer** dan **User**.

---

## 2. Target Sistem Akhir

Target implementasi dibagi menjadi dua tingkat agar pembangunan dapat dilakukan bertahap.

### 2.1 Target utama yang wajib selesai

```text
ESP32-CAM (Akuisisi Frame VGA/QVGA)
    │
    │ Wi-Fi (MJPEG / Raw Binary Stream)
    ▼
ML Server / Laptop
    │
    ▼
YOLOv8n
    │  deteksi keberadaan tangan + bounding box
    ▼
Crop / ROI Tangan
    │
    ▼
MediaPipe Hands
    │  ekstraksi 21 hand landmarks (x, y, z)
    ▼
Normalisasi Landmark
    │  vektor fitur spasial (63 float)
    ▼
1D-CNN
    │  ekstraksi representasi spasial gesture
    ▼
LSTM
    │  analisis urutan frame / temporal sequence
    ▼
Gesture Class (14 Kelas)
    │
    ▼
Temporary Buffer / Sentence Builder
    │
    ▼
Teks Kalimat (Subjek + Predikat)
    │
    ▼
Text-to-Speech (TTS)
    │
    ▼
Keluaran Suara
```

Website dan database SQLite berjalan sebagai lapisan manajemen dan antarmuka yang berkomunikasi dengan ML Engine.

### 2.2 Pengembangan lanjutan

Setelah alur utama stabil, sistem dipindahkan ke edge deployment:

```text
ESP32-CAM
    │
    ├── Wi-Fi → ML Server Laptop (Development/Training)
    │
    └── Deployment Lapangan → SBC (Raspberry Pi 5 4GB)
                               │
                               ├── YOLOv8n (ONNX/TFLite)
                               ├── MediaPipe Hands
                               ├── 1D-CNN + LSTM
                               └── FastAPI + SQLite
```

Komunikasi UART berkecepatan tinggi, edge computing/SBC, optimasi model, dan pemisahan layanan dapat dianggap sebagai pengembangan lanjutan. Komponen tersebut **tidak menjadi prasyarat untuk menyelesaikan pipeline utama**.

---

## 3. Pembagian Tugas Setiap Komponen ML

| Komponen | Tugas utama | Input | Output |
|---|---|---|---|
| ESP32-CAM | Akuisisi citra real-time | Lingkungan/gestur tangan | Frame gambar (VGA/QVGA) |
| YOLO (YOLOv8n) | Deteksi objek tangan & trigger | Frame citra | Bounding box tangan, confidence |
| ROI/Crop | Membatasi area proses | Frame + bbox YOLO | Crop area tangan |
| MediaPipe | Ekstraksi struktur tangan | Crop area tangan | 21 landmark tangan (x, y, z) |
| 1D-CNN | Ekstraksi fitur spasial landmark | 63 koordinat landmark ternormalisasi | Vektor fitur spasial gesture |
| LSTM | Memproses urutan temporal frame | Sekuens vektor fitur spasial | Prediksi kelas gesture (14 kelas) |
| Sentence Builder | Menyusun kata terstruktur | Hasil klasifikasi gesture | Kalimat sementara (S + P) |
| TTS | Mengubah teks menjadi audio | Kalimat | Suara |

### Catatan penting tentang Alur Ekstraksi Fitur (YOLO vs MediaPipe vs CNN)

1. **YOLO** difungsikan sebagai **hand presence detector** yang sangat cepat untuk memastikan tangan berada di dalam frame sebelum diproses lebih lanjut.
2. **MediaPipe Hands** mengubah citra piksel menjadi struktur geometris **21 titik landmark 3D**. Dengan cara ini, citra tangan yang berukuran puluhan ribu piksel tereduksi secara drastis menjadi **63 angka float**.
3. **1D-CNN** mempelajari korelasi spasial antar sendi tangan dari 63 angka float tersebut (bukan memproses ulang piksel citra secara berat).
4. **LSTM** menganalisis perubahan vektor fitur tersebut sepanjang waktu untuk mengenali gestur dinamis maupun stabilisasi gestur statis.

Dengan pemisahan ini:
- Komputasi di SBC menjadi sangat ringan (>90% lebih hemat dibanding 2D-CNN citra mentah).
- Latensi per frame dapat ditekan hingga di bawah 40 ms.
- Seluruh komponen proposal (YOLO, MediaPipe, CNN, LSTM) tetap terpenuhi secara proporsional dan elegan.

---

## 4. Kelas Gesture Awal

Mengikuti tabel gesture pada proposal saat ini, dataset gesture recognition menggunakan 14 kelas:

### Subjek

1. `saya`
2. `kamu`
3. `anda`
4. `kami`
5. `kita`
6. `dia`
7. `mereka`

### Predikat

8. `makan`
9. `minum`
10. `tidur`
11. `belajar`
12. `bekerja`
13. `berjalan`
14. `membaca`

> **Aturan sinkronisasi:** apabila hasil revisi dosen kemudian mengurangi atau mengganti kelas gesture, seluruh bagian berikut harus diperbarui bersama: dataset, label map, konfigurasi training, model, confusion matrix, sentence builder, UI, dan dokumentasi.

---

## 5. Strategi Dataset

Dataset sebaiknya dipisahkan secara logis menjadi minimal dua kebutuhan.

### 5.1 Dataset YOLO

Tujuan dataset ini hanya untuk mengenali keberadaan dan lokasi tangan.

```text
dataset/yolo/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Kelas awal:

```text
0 = hand
```

Label YOLO berisi koordinat bounding box sesuai format YOLO.

### 5.2 Dataset Gesture Recognition

Tujuan dataset ini untuk mengenali 14 kelas gesture.

```text
dataset/gesture/
├── saya/
├── kamu/
├── anda/
├── kami/
├── kita/
├── dia/
├── mereka/
├── makan/
├── minum/
├── tidur/
├── belajar/
├── bekerja/
├── berjalan/
└── membaca/
```

Untuk CNN, data dapat berupa frame/crop tangan.

Untuk LSTM, data harus memiliki struktur urutan, misalnya:

```text
sequence_0001/
├── frame_001.jpg
├── frame_002.jpg
├── ...
└── frame_020.jpg
```

Atau yang lebih efisien pada tahap training:

```text
sequence_id → [feature_1, feature_2, ..., feature_T]
label       → gesture_class
```

### 5.3 Variasi data

Pengumpulan data mempertahankan variasi yang sudah ditetapkan pada proposal:

- posisi tangan;
- jarak tangan terhadap kamera;
- sudut pandang;
- kondisi pencahayaan;
- latar belakang;
- ukuran tangan;
- variasi kecil orientasi telapak dan jari;
- variasi kecepatan gerakan untuk data temporal.

### 5.4 Strategi jumlah data awal

Agar pembangunan tidak terlalu berat pada tahap awal, dataset dapat dibuat bertahap.

**Prototype awal:** gunakan jumlah data yang cukup untuk menguji pipeline, bukan langsung mengejar dataset final berukuran besar.

Contoh target awal yang dapat digunakan sebagai baseline internal:

- YOLO: sekitar 1.000–2.000 gambar total untuk prototipe awal.
- Gesture classifier: sekitar 200–300 frame/class sebagai tahap awal.
- Sequence LSTM: sekitar 30–50 sequence/class sebagai tahap awal.

Angka tersebut adalah **target kerja**, bukan angka wajib pada proposal. Jumlah akhir ditentukan berdasarkan kualitas data, hasil evaluasi, dan waktu pengerjaan.

---

## 6. Aturan Pengumpulan Dataset

Agar dataset mudah diperbaiki, setiap data sebaiknya memiliki metadata minimal:

```text
sample_id
class_name
category
source
capture_device
lighting_condition
camera_distance
view_angle
sequence_id
created_at
```

Untuk sistem yang lebih sederhana, metadata tersebut tidak harus langsung disimpan seluruhnya ke MongoDB. Pada tahap awal dapat disimpan dalam struktur folder dan file CSV/JSON, kemudian disinkronkan ke database pada tahap implementasi website.

### Prosedur pengambilan data

```text
Pilih gesture
      ↓
Pastikan posisi tangan benar
      ↓
Capture melalui webcam / ESP32-CAM
      ↓
Periksa kualitas frame
      ↓
Simpan label
      ↓
Validasi data
      ↓
Masukkan ke dataset
```

Data yang buram, tertutup sebagian, salah gesture, atau tidak relevan harus dipindahkan ke folder review dan tidak langsung digunakan untuk training.

---

## 7. Pipeline Preprocessing

Pipeline preprocessing harus dibuat konsisten antara training dan inferensi.

### 7.1 YOLO

```text
Image
  ↓
Resize / letterbox oleh pipeline YOLO
  ↓
YOLO inference/training
  ↓
Bounding box hand
```

### 7.2 Gesture recognition

```text
Frame (dari ESP32-CAM)
  ↓
YOLOv8n detect hand
  ↓
Crop ROI bounding box
  ↓
MediaPipe Hand Landmarks
  ↓ (21 titik 3D: x, y, z)
Normalisasi Landmark (Wrist centered + scale invariant)
  ↓ (Vektor 63 float)
1D-CNN (Ekstraksi korelasi spasial antar sendi)
  ↓ (Feature vector spasial 64/128-d)
Sequence Buffer (10–15 frame)
  ↓
LSTM (Analisis sekuens temporal)
  ↓
Dense + Softmax
  ↓
Gesture Class (14 kelas)
```

### 7.3 Normalisasi Landmark

Landmark MediaPipe dinormalisasi sebelum digunakan sebagai input 1D-CNN. Tujuannya agar model kebal terhadap translasi posisi tangan di layar dan variasi jarak tangan ke kamera.

Prosedur normalisasi:

```text
21 landmark × (x, y, z) = 63 nilai mentah
        ↓
1. Translasi: Kurangi semua koordinat terhadap koordinat wrist (landmark 0)
   (P_i' = P_i - P_wrist, sehingga koordinat wrist menjadi (0, 0, 0))
        ↓
2. Skala: Bagi seluruh koordinat dengan jarak Euclidean maksimum dari wrist ke ujung jari tengah (landmark 12)
        ↓
Vektor fitur spasial ternormalisasi (63 float dalam rentang [-1.0, 1.0])
```

Normalisasi ini wajib digunakan secara identik pada proses training maupun inferensi real-time.

---

## 8. Perancangan 1D-CNN (Spatial Landmark Encoder)

CNN digunakan untuk mempelajari pola korelasi spasial antar sendi tangan dari vektor landmark yang sudah dinormalisasi.

### Mengapa 1D-CNN, Bukan 2D-CNN Citra Piksel?

1. **Reduksi Data Drastis:** Memproses 63 nilai float vs citra $128 \times 128 \times 3$ (49.152 piksel) menghemat pemrosesan data hingga **780x lipat**.
2. **Bebas Distorsi Latar Belakang:** Citra piksel rentan terganggu oleh perubahan warna kulit, latar belakang ruangan, dan intensitas cahaya. Vektor landmark hanya memuat struktur geometris tangan.
3. **Efisiensi Komputasi Ekstrem:** 1D-CNN dapat dieksekusi di CPU SBC Raspberry Pi 5 hanya dalam waktu **< 1 ms per frame**, menyisakan kapasitas CPU yang sangat lega untuk tugas lain.

### Arsitektur 1D-CNN Ringan

```text
Input: Vector (63, 1) atau (21, 3)
   ↓
Conv1D (32 filters, kernel_size=3, padding='same', activation='relu')
   ↓
Batch Normalization
   ↓
MaxPool1D (pool_size=2)
   ↓
Conv1D (64 filters, kernel_size=3, padding='same', activation='relu')
   ↓
GlobalAveragePooling1D / Flatten
   ↓
Dense (64 units, activation='relu')
   ↓
Dropout (0.3)
   ↓
Output: Spatial Feature Vector (64-dimensi)
```

### Output 1D-CNN ke LSTM

Setiap frame $t$ menghasilkan 1 vektor fitur spasial berdimensi 64:

```text
Frame t1 → 1D-CNN → feature_1 (64-dim)
Frame t2 → 1D-CNN → feature_2 (64-dim)
...
Frame tT → 1D-CNN → feature_T (64-dim)
```

Vektor-vektor ini dikumpulkan dalam sequence buffer berukuran $T$ (misal $T = 12$ frame) untuk diumpankan ke LSTM.

---

## 9. Perancangan LSTM

LSTM digunakan untuk memproses dinamika perubahan gestur sepanjang waktu berdasarkan rangkaian fitur spasial dari 1D-CNN.

### Pipeline LSTM

```text
Sekuens Fitur 1D-CNN (Shape: [Batch, T=12, Features=64])
        ↓
LSTM Layer 1 (64 units, return_sequences=False)
        ↓
Dropout (0.3)
        ↓
Dense (32 units, activation='relu')
        ↓
Dense (14 units, activation='softmax')
        ↓
Predicted Gesture Class (14 Kelas) + Confidence Score
```

### Parameter awal yang dapat diuji

Sebagai eksperimen awal:

- sequence length: 10–20 frame;
- batch size: 8–32;
- 1–2 LSTM layer;
- dropout untuk mengurangi overfitting;
- epoch tidak ditetapkan secara permanen; dihentikan menggunakan validation performance/early stopping bila tersedia.

Parameter final ditentukan berdasarkan hasil pengujian model, bukan hanya berdasarkan angka awal.

### Peran LSTM untuk gesture statis

Sebagian gesture pada tabel proposal dapat memiliki karakteristik yang dominan statis. Pada kondisi tersebut, LSTM tidak perlu dipaksakan untuk “menciptakan” gerakan dinamis. LSTM dapat digunakan untuk memanfaatkan beberapa frame berurutan sebagai dasar stabilisasi prediksi dan membedakan perubahan gesture dari waktu ke waktu.

---

## 10. Strategi Training Model

Training dilakukan bertahap, jangan langsung melatih seluruh pipeline sekaligus.

### Tahap A — YOLO

```text
Dataset YOLO
    ↓
Training YOLO hand detector
    ↓
Validation
    ↓
Test
    ↓
Simpan best model
```

Hasil yang dibutuhkan:

```text
models/yolo/
├── best.pt
└── metadata.json
```

### Tahap B — CNN

```text
Dataset Koordinat Landmark Ternormalisasi (63-d)
    ↓
1D-CNN Training (Spatial Feature Extractor)
    ↓
Validation
    ↓
Confusion Matrix
    ↓
Simpan Model & Export (ONNX / TFLite)
```

### Tahap C — 1D-CNN + LSTM

```text
Dataset Sekuens Landmark (T frame × 63-d)
    ↓
1D-CNN Feature Extraction per Frame
    ↓
Penyusunan Sekuens Fitur (T × 64-d)
    ↓
LSTM Training (Temporal Classifier)
    ↓
Validation & Evaluation (F1-score, Loss)
    ↓
Simpan Best Model & Export (ONNX / TFLite)
```

### Strategi mempercepat proses training

1. **Ekstraksi Fitur Landmark Terlebih Dahulu:** Jalankan MediaPipe pada seluruh video/dataset satu kali saja, lalu simpan koordinat landmark dalam bentuk file `.npy` / `.csv`. Jangan menjalankan MediaPipe berulang-ulang di setiap epoch training!
2. Gunakan early stopping pada validation loss untuk mencegah overfitting.
3. Simpan checkpoint setiap epoch terbaik.
4. Pisahkan proses ekstraksi feature dengan proses training LSTM.
5. Gunakan laptop atau Google Colab untuk training; jangan melakukan training di SBC.

---

## 11. Penyimpanan Artefak Model

Setiap model harus memiliki identitas yang jelas dan format ekspor yang siap dieksekusi di edge:

```text
models/
├── yolo/
│   ├── hand_detector_v001.pt
│   └── hand_detector_v001.onnx      # Format siap deploy ke SBC
├── cnn/
│   ├── gesture_1dcnn_v001.keras
│   └── gesture_1dcnn_v001.onnx     # Format ringan
├── lstm/
│   ├── gesture_lstm_v001.keras
│   └── gesture_lstm_v001.tflite    # Format TFLite runtime
└── exported/
```

Metadata model:

```json
{
  "model_name": "gesture_lstm_v001",
  "model_type": "1dcnn_lstm",
  "dataset_version": "gesture-v001",
  "classes": 14,
  "sequence_length": 12,
  "created_at": "YYYY-MM-DD",
  "accuracy": 0.0,
  "precision": 0.0,
  "recall": 0.0,
  "f1_score": 0.0,
  "status": "inactive"
}
```

Nilai performa pada contoh di atas harus diisi dari hasil pengujian sebenarnya.

---

## 12. Model Management

Model Management merupakan fitur penting dari rancangan sistem karena sistem tidak hanya memiliki satu model selama proses pengembangan.

Fungsi utama:

```text
Upload model (.onnx / .tflite) via Web
        ↓
Register model & metadata
        ↓
Validasi arsitektur input/output
        ↓
Status: candidate
        ↓
Developer memilih deploy (Hot-Swap)
        ↓
Status: active
```

Status model yang disarankan:

```text
candidate
active
inactive
archived
failed
```

Hanya satu kombinasi model yang boleh menjadi konfigurasi aktif untuk pipeline produksi pada satu waktu.

---

## 13. Arsitektur ML Engine

Direktori pemrosesan machine learning dibuat modular dan terpisah dari kode web:

```text
ml-engine/
├── app.py
├── config/
│   ├── settings.py
│   └── labels.json
├── pipelines/
│   ├── yolo_detector.py        # YOLOv8n hand detector
│   ├── mediapipe_processor.py  # 21 Hand landmarks extractor
│   ├── landmark_normalizer.py  # Normalisasi translasi & skala
│   ├── cnn_encoder.py          # 1D-CNN spatial encoder
│   ├── lstm_predictor.py       # LSTM temporal sequence predictor
│   ├── sentence_builder.py     # State machine (S + P)
│   └── tts_engine.py           # Piper / eSpeak / gTTS
├── inference/
│   ├── realtime.py             # Main inference loop
│   └── session.py
├── training/                   # Standalone scripts (Laptop/Colab)
│   ├── extract_landmarks.py    # Pre-generate landmark dataset
│   ├── train_yolo.py
│   ├── train_1dcnn.py
│   ├── train_lstm.py
│   └── export_onnx.py          # Export ke format edge
├── dataset/
├── models/
├── logs/
└── requirements.txt
```

Prinsipnya: kode training dieksekusi secara mandiri (*standalone CLI*) di Laptop/Colab dan tidak bercampur dengan runtime inferensi real-time.

---

## 14. Alur Inferensi Real-Time

### 14.1 Frame processing

```text
ESP32-CAM mengirim frame (MJPEG / Stream)
        ↓
ML Server menerima frame
        ↓
YOLOv8n inference (Hand presence check)
        ↓
Ada object hand?
   ├── Tidak → frame berikutnya (skip)
   └── Ya
        ↓
Ambil bounding box (tambahkan margin/padding ~15%)
        ↓
Crop ROI tangan
        ↓
MediaPipe Hands
        ↓
21 landmarks berhasil diekstrak?
   ├── Tidak → frame berikutnya (skip)
   └── Ya
        ↓
Normalisasi Landmark (Wrist center + Scale)
        ↓
1D-CNN spatial feature extraction (vektor 64-d)
        ↓
Masukkan fitur ke Sequence Buffer (FIFO, misal length=12)
        ↓
Sequence Buffer penuh?
   ├── Belum → tunggu frame berikutnya
   └── Ya
        ↓
LSTM Predictor
        ↓
Gesture prediction + confidence score
        ↓
Stabilization / majority voting filter
        ↓
Gesture valid & stabil?
   ├── Tidak → jangan masukkan ke kalimat
   └── Ya
        ↓
Sentence Builder (State machine: Subjek + Predikat)
        ↓
Teks ditampilkan di UI
        ↓
TTS (Audio output saat kalimat selesai)
```

---

## 15. Stabilization Prediksi

Sistem real-time tidak seharusnya memasukkan setiap hasil frame sebagai kata baru. Tanpa stabilisasi, satu gesture yang dipertahankan selama beberapa frame dapat menghasilkan:

```text
saya saya saya saya saya
```

Padahal yang dimaksud hanya satu kata `saya`.

Karena itu ditambahkan mekanisme **prediction stabilization**.

Contoh aturan:

```text
Prediction:
Saya, Saya, Saya, Saya, Saya
        ↓
Majority / temporal consistency
        ↓
Accepted: Saya
```

Aturan dapat menggunakan:

- jumlah frame berturut-turut;
- confidence minimum;
- majority vote;
- cooldown setelah sebuah kata diterima;
- perubahan gesture sebagai pemicu penerimaan kata berikutnya.

Nilai threshold tidak ditetapkan sebagai angka mutlak dari awal. Nilai tersebut dituning menggunakan hasil testing.

---

## 16. Sentence Builder

Karena kelas gesture dibagi menjadi Subjek dan Predikat, pembentukan kalimat dapat menggunakan aturan sederhana.

### Struktur dasar

```text
SUBJEK + PREDIKAT
```

Contoh:

```text
saya + makan     → Saya makan.
kami + belajar   → Kami belajar.
dia + bekerja    → Dia bekerja.
anda + membaca   → Anda membaca.
```

### State machine sederhana

```text
EMPTY
  ↓
SUBJECT_RECEIVED
  ↓
PREDICATE_RECEIVED
  ↓
SENTENCE_READY
  ↓
TTS
  ↓
RESET
```

### Aturan tambahan

- Subjek baru menggantikan subjek lama jika predikat belum diterima.
- Predikat tidak boleh diproses sebagai subjek.
- Setelah pasangan subjek + predikat lengkap, sistem dapat menjalankan TTS.
- Temporary buffer kemudian dikosongkan atau dipindahkan ke history.

Aturan ini mengikuti cakupan kalimat sederhana pada proposal dan tidak mencoba menjadi parser bahasa alami penuh.

---

## 17. Text-to-Speech

Alur TTS:

```text
Gesture class
    ↓
Word
    ↓
Sentence Builder
    ↓
Final sentence
    ↓
TTS engine
    ↓
Audio output
```

Sistem tidak perlu menghasilkan suara pada setiap perubahan kecil frame. TTS lebih aman dipicu ketika kalimat sudah dianggap valid/selesai.

Event yang dapat digunakan:

- sentence complete;
- tombol `Speak`;
- timeout setelah gesture terakhir;
- command reset/new sentence.

---

## 18. Komunikasi ESP32-CAM ↔ ML Server

Tahap pertama menggunakan komunikasi jaringan Wi-Fi sesuai rancangan proposal.

### Payload minimal

```json
{
  "device_id": "esp32cam-001",
  "timestamp": 0,
  "frame_id": 123,
  "image_format": "jpeg",
  "image": "<binary/base64 sesuai desain transport>"
}
```

Untuk implementasi real-time, transmisi binary lebih efisien daripada base64. Struktur final payload harus didokumentasikan sebelum integrasi.

### Respons server

```json
{
  "frame_id": 123,
  "detected": true,
  "gesture": "saya",
  "confidence": 0.0,
  "sentence": "Saya makan.",
  "device_status": "online"
}
```

Angka confidence pada contoh harus diisi oleh sistem aktual.

---

## 19. Opsi Pengembangan SBC / Raspberry Pi

Dokumen arsitektur tambahan mengusulkan komputasi terdistribusi menggunakan SBC seperti Raspberry Pi. Konsep tersebut dapat dimasukkan sebagai **extension**, bukan pengganti pipeline utama.

### Mode server laptop

```text
ESP32-CAM → Wi-Fi → Laptop → ML Pipeline
```

### Mode edge/SBC

```text
ESP32-CAM → Wi-Fi/UART → Raspberry Pi → ML Pipeline
```

Manfaat yang dapat diuji:

- mengurangi ketergantungan pada laptop;
- memisahkan device acquisition dan ML processing;
- membangun edge inference yang lebih mandiri;
- membuka peluang sistem berjalan sebagai perangkat khusus.

### UART

Rancangan tambahan menyebut penggunaan UART dan baud rate tinggi sebagai opsi untuk transmisi JPEG. Implementasi UART sebaiknya dilakukan **setelah pipeline Wi-Fi stabil** karena perubahan media komunikasi tidak perlu dilakukan bersamaan dengan perubahan model ML.

---

## 20. Cloud GPU sebagai Opsi Training

Laptop digunakan untuk development, debugging, preprocessing, dan pengujian. Ketika training berat, model dapat dipindahkan ke cloud GPU.

```text
Local dataset
    ↓
Prepare dataset
    ↓
Export package
    ↓
Cloud GPU
    ↓
Training
    ↓
Evaluation
    ↓
Download best model
    ↓
Local inference test
```

Model hasil training harus selalu diuji kembali di lingkungan lokal sebelum dianggap siap digunakan.

---

## 21. Website dan Database (SQLite)

Website mempertahankan dua peran yang sudah dirancang dengan backend FastAPI yang ringan.

### Developer

Developer dapat:

- melihat dashboard operasional & metrik performa;
- mengelola dataset dan ekspor data koordinat landmark;
- mengunggah model baru (.onnx / .tflite) dan melihat metrik evaluasi;
- mengaktifkan/mengganti model (Hot-Swap Deployment) atau rollback model;
- melakukan live testing dengan overlay visual landmark;
- memantau status perangkat ESP32-CAM.

### User

User hanya menggunakan:

- dashboard real-time yang bersih;
- visualisasi frame dan nama gesture;
- confidence score;
- kalimat yang tersusun (Subjek + Predikat);
- tombol pemicu suara (Speak) dan tombol reset kalimat;
- status koneksi ESP32-CAM.

### Struktur Tabel SQLite yang Disarankan

Menggunakan **SQLite** (`hand_gesture.db`) yang tertanam langsung (*in-process*) di Python, tanpa daemon terpisah, zero memory overhead, dan mendukung query kolom JSON:

```text
users (id, username, password_hash, role, created_at)
  └── akun dan hak akses (developer / user)

datasets (id, name, version, sample_count, metadata_json, created_at)
  └── riwayat versi dataset

models (id, name, type, version, accuracy, f1_score, file_path, status, created_at)
  └── metadata dan path file model (.onnx / .tflite)

detections (id, timestamp, detected_gesture, confidence, sentence_output, raw_data_json)
  └── riwayat log deteksi

devices (id, device_name, ip_address, status, last_heartbeat)
  └── status koneksi perangkat IoT ESP32-CAM
```

Keuntungan SQLite:
1. **0 MB RAM Standby:** Tidak seperti MongoDB yang mengunci ratusan MB RAM.
2. **Satu File Portabel:** File `hand_gesture.db` dapat di-copy langsung untuk backup tanpa prosedur dump rumit.
3. **Mendukung JSON:** Struktur dinamis tetap dapat disimpan dalam kolom bertipe JSON.

---

## 22. Dataset Management pada Website

Fitur yang dipertahankan dari proposal:

```text
Upload video/image
Capture webcam / ESP32-CAM
Pilih label gesture (14 kelas)
Preview visual
```

Dapat ditambahkan:

```text
Validasi kelengkapan 21 landmark
Hapus sample blur / rusak
Ekspor dataset ke format koordinat (.npy / .csv)
Dataset versioning (gesture-v001, v002)
```

Alur:

```text
Input data (video/gambar)
   ↓
Ekstraksi MediaPipe Landmarks
   ↓
Validasi kelengkapan 21 titik
   ↓
Simpan koordinat ternormalisasi (63 float)
   ↓
Dataset siap training
```

Data yang gagal diekstrak landmark-nya tidak dimasukkan ke dalam training set.

---

## 23. Manajemen Hasil Training & Evaluasi Model

Training model dilakukan **secara mandiri di Laptop atau Google Colab** menggunakan skrip Python (`train_1dcnn.py`, `train_lstm.py`), bukan sebagai background job berat di dalam web server.

Halaman Web difokuskan untuk:

- melihat riwayat training run (loss curve, accuracy curve, confusion matrix);
- membandingkan performa antar versi model;
- mengunggah artefak model siap pakai (`.onnx` atau `.tflite`);
- mengaktifkan model terpilih untuk dipakai inferensi real-time.

```text
Laptop / Colab (Training Script)
       ↓
Evaluasi & Export (.onnx / .tflite)
       ↓
Upload ke Web Dashboard
       ↓
Validasi File
       ↓
Model Registry
       ↓
Hot-Swap Aktif
```

---

## 24. Live Testing

Halaman Testing mempertahankan tampilan proposal:

- video/capture;
- bounding box;
- nama gesture;
- confidence;
- opsi menambahkan data jika prediksi salah.

Dapat diperluas dengan informasi debugging:

```text
YOLO confidence
MediaPipe detected
Landmark count
CNN feature status
LSTM sequence length
Final confidence
Processing time
FPS
```

Mode debug sebaiknya hanya tersedia untuk Developer agar tampilan User tetap sederhana.

---

## 25. Model Registry dan Deployment

Deployment tidak cukup hanya menyimpan file `.pt` atau `.keras`. Sistem harus menyimpan hubungan antara model, dataset, dan hasil evaluasi.

Contoh:

```text
Dataset v003
    ↓
YOLO v002
    ↓
CNN v004
    ↓
LSTM v003
    ↓
Evaluation
    ↓
Pipeline Candidate v005
    ↓
Developer deploy
    ↓
Pipeline Active
```

Dengan konsep tersebut, ketika terjadi regresi performa, pipeline sebelumnya dapat diaktifkan kembali.

---

## 26. Logging dan Observability

Log minimal yang perlu disimpan selama development:

```text
timestamp
frame_id
device_id
yolo_detected
yolo_confidence
mediapipe_status
cnn_status
lstm_prediction
prediction_confidence
sequence_length
processing_time
sentence_buffer
tts_status
```

Tujuannya bukan hanya debugging, tetapi mencari sumber error.

Contoh:

```text
YOLO gagal
    → masalah deteksi/tidak ada ROI

YOLO berhasil + MediaPipe gagal
    → masalah crop/quality/pose

MediaPipe berhasil + classifier salah
    → masalah dataset/model

Classifier benar + kalimat salah
    → masalah Sentence Builder

Kalimat benar + tidak ada suara
    → masalah TTS
```

---

## 27. Strategi Error Handling

Sistem harus tetap berjalan ketika salah satu frame gagal diproses.

### Kasus 1 — YOLO tidak menemukan tangan

```text
Tidak ada bbox
   ↓
Skip frame
   ↓
Ambil frame berikutnya
```

### Kasus 2 — MediaPipe tidak menemukan landmark

```text
BBox ada
   ↓
MediaPipe gagal
   ↓
Jangan memasukkan feature invalid
   ↓
Ambil frame berikutnya
```

### Kasus 3 — Confidence rendah

```text
Prediction confidence < threshold
   ↓
Status uncertain
   ↓
Tidak memasukkan kata ke buffer
```

### Kasus 4 — Connection ESP32-CAM terputus

```text
Heartbeat timeout
   ↓
Device = offline
   ↓
UI menampilkan status offline
   ↓
ML engine berhenti menunggu frame baru
```

---

## 28. Pengujian Model

Proposal sudah menetapkan accuracy, precision, recall, F1-score, dan confusion matrix untuk pengujian model. Implementasi teknis dapat membaginya menjadi dua level agar hasil evaluasi lebih jelas.

### 28.1 YOLO object detection

Metrik yang relevan:

- precision;
- recall;
- mAP@0.5;
- mAP@0.5:0.95 bila diperlukan;
- inference time/FPS.

Fokusnya adalah keberhasilan menemukan objek `hand` dan bounding box yang benar.

### 28.2 Gesture classification

Metrik:

- accuracy;
- precision;
- recall;
- F1-score;
- confusion matrix.

Fokusnya adalah kemampuan membedakan kelas `saya`, `kamu`, ..., `membaca`.

### 28.3 End-to-end system

Tambahan pengujian sistem:

- response time;
- FPS;
- latency ESP32-CAM → server;
- latency inference;
- waktu pembentukan kalimat;
- keberhasilan TTS;
- kestabilan koneksi.

---

## 29. Pengujian Kondisi Lingkungan

Kondisi yang sudah tercantum pada proposal harus diuji secara nyata:

| Variabel | Contoh kondisi |
|---|---|
| Pencahayaan | cukup, lebih redup, lebih terang |
| Jarak | dekat, sedang, jauh |
| Sudut | frontal, miring kiri, miring kanan |
| Posisi | tengah, agak kiri, agak kanan |
| Background | sederhana, lebih kompleks |
| Kecepatan | gesture stabil, gerakan lebih cepat |

Tujuan pengujian ini adalah mengetahui kondisi ketika model mulai mengalami penurunan performa.

---

## 30. Blackbox Testing

Fitur yang perlu diuji minimal:

| Fitur | Input | Expected output |
|---|---|---|
| Login Developer | kredensial valid | masuk dashboard Developer |
| Login User | kredensial valid | masuk dashboard User |
| Upload dataset | gambar + label | data tersimpan |
| Capture | webcam/camera | frame tersimpan |
| Training | dataset + parameter | training berjalan |
| Model Management | model candidate | model dapat diaktifkan |
| Live Testing | gestur tangan | gesture/confidence tampil |
| Sentence Builder | subjek + predikat | kalimat tersusun |
| TTS | kalimat | audio diputar |
| Device status | ESP32 online/offline | status sesuai kondisi |
| Detection history | hasil deteksi | history tersimpan |

Kolom tambahan untuk laporan:

```text
Actual Result
Status
Evidence / Screenshot
```

---

## 31. Testing Data Leakage

Dataset harus dibagi menjadi train, validation, dan test tanpa membuat sequence dari sumber yang sama tersebar secara tidak semestinya ke seluruh split.

Contoh masalah:

```text
Video A
 ├── frame 001 → train
 ├── frame 002 → train
 └── frame 003 → test
```

Kondisi tersebut berpotensi membuat test terlalu mirip dengan training.

Lebih aman:

```text
Video A → train
Video B → validation
Video C → test
```

Aturan split ini harus diperhatikan terutama pada data LSTM yang berasal dari video berurutan.

---

## 32. Versioning Dataset

Gunakan versi dataset sederhana:

```text
gesture-v001
 gesture-v002
 gesture-v003
```

Setiap versi mencatat:

```text
classes
jumlah sample/class
tanggal
perubahan
sumber data
```

Contoh:

```text
gesture-v002
- tambah 80 sample "saya"
- tambah 60 sample "makan"
- hapus 15 frame blur
- perbaiki label "dia"
```

Dengan demikian, ketika model berubah, sumber perubahan dapat dilacak.

---

## 33. Struktur Repository yang Direkomendasikan

Mengambil ide modular dari dokumen arsitektur tambahan, repository dapat disusun sebagai berikut:

```text
hand-gesture-iot/
├── iot-node/
│   ├── esp32-cam/
│   └── README.md
│
├── ml-engine/
│   ├── pipelines/
│   ├── inference/
│   ├── training/
│   ├── preprocessing/
│   ├── models/
│   └── README.md
│
├── web-dashboard/
│   ├── app/
│   ├── templates/
│   ├── static/
│   └── README.md
│
├── dataset/
│   ├── yolo/
│   └── gesture/
│
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── model-card.md
│
├── scripts/
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

Folder dapat disesuaikan dengan framework web yang akhirnya digunakan.

---

## 34. API Contract

Sebelum integrasi IoT dan website, struktur endpoint perlu ditetapkan.

Contoh endpoint (FastAPI):

```text
POST /api/inference/frame         # Menerima frame dari ESP32-CAM / Web
GET  /api/device/status          # Status koneksi IoT (online/offline)
GET  /api/models/active          # Informasi model aktif
GET  /api/detections/history     # Riwayat log deteksi dari SQLite
POST /api/dataset/upload         # Upload data sample/video
POST /api/models/upload          # Upload artefak model (.onnx / .tflite)
POST /api/models/{id}/deploy     # Mengaktifkan model (Hot-Swap)
POST /api/models/{id}/rollback   # Membatalkan model aktif
POST /api/sentence/reset         # Reset buffer kalimat
POST /api/tts/speak              # Trigger keluaran suara manual/otomatis
```

Endpoint final dapat disesuaikan. Yang penting adalah format request/response terdokumentasi sejak awal.

---

## 35. Security Dasar

Walaupun fokus proyek berada pada machine learning, aplikasi tetap perlu memiliki perlindungan dasar:

- autentikasi Developer dan User;
- authorization berbasis role;
- validasi upload file;
- pembatasan tipe dan ukuran file;
- sanitasi nama file;
- validasi parameter training;
- pembatasan endpoint training agar tidak dapat digunakan User;
- penyimpanan secret di environment variable;
- jangan menyimpan password plaintext;
- pencatatan aktivitas penting Developer.

Dataset dan model juga perlu dianggap sebagai aset sistem. File `.pt`, `.keras`, `.h5`, dan dataset mentah tidak boleh dianggap sama dengan file biasa yang selalu aman untuk diunggah tanpa validasi.

---

## 36. Git dan Model Storage

Model yang besar tidak ideal dimasukkan langsung ke repository Git biasa. Opsi yang dapat digunakan:

```text
Git LFS
atau
GitHub Releases / artifact storage
atau
object storage
```

Repository menyimpan:

```text
code
config
metadata
training script
README
```

Sedangkan file model besar disimpan dengan mekanisme storage yang sesuai.

---

## 37. Dashboard Developer

Dashboard Developer mempertahankan elemen yang ada di proposal:

```text
Dataset Count
Gesture Classes
Active Model
Accuracy
ESP32 Status
Training Accuracy
Training Loss
```

Pengembangan dapat menambahkan:

```text
Last Training
Training Duration
YOLO mAP
CNN Accuracy
LSTM F1
Current Pipeline Version
Last Deployment
Recent Errors
```

Dashboard sebaiknya menampilkan data yang berasal dari database/log aktual, bukan angka dummy setelah sistem mulai diuji.

---

## 38. Dashboard User

Dashboard User dibuat sederhana.

```text
┌───────────────────────────────┐
│ ESP32-CAM: ONLINE              │
├───────────────────────────────┤
│ Camera Preview                │
│                               │
│       [ Hand / BBox ]         │
│                               │
├───────────────────────────────┤
│ Gesture   : saya              │
│ Confidence: 0.00              │
│ Sentence  : Saya makan.       │
├───────────────────────────────┤
│ [ Speak ] [ Reset ]            │
└───────────────────────────────┘
```

Tampilan akhir harus tetap mengikuti kebutuhan aksesibilitas dan kemudahan penggunaan.

---

## 39. Tahapan Pembangunan yang Disarankan

Urutan pembangunan berikut dibuat agar debugging tidak terlalu sulit.

### Fase 1 — Camera acquisition

**Target:** ESP32-CAM dapat mengambil gambar dan mengirim frame.

Checklist:

- kamera hidup;
- Wi-Fi terhubung;
- frame dapat diterima server;
- frame dapat disimpan/ditampilkan.

### Fase 2 — YOLO hand detector

**Target:** server dapat mendeteksi `hand` dan menampilkan bounding box.

Checklist:

- dataset YOLO siap;
- training berhasil;
- inference berhasil;
- bbox stabil;
- confidence tercatat.

### Fase 3 — MediaPipe

**Target:** ROI dari YOLO dapat diproses dan menghasilkan 21 landmark.

Checklist:

- crop valid;
- landmark terdeteksi;
- koordinat divisualisasikan;
- frame gagal ditangani tanpa crash.

### Fase 4 — CNN gesture recognition

**Target:** CNN dapat membedakan kelas gesture dari data yang telah disiapkan.

Checklist:

- train/val/test tersedia;
- training selesai;
- confusion matrix tersedia;
- model disimpan.

### Fase 5 — LSTM

**Target:** sequence feature dapat diproses untuk menghasilkan prediksi temporal.

Checklist:

- sequence builder selesai;
- sequence length konsisten;
- LSTM dapat training;
- evaluasi tersedia.

### Fase 6 — Real-time inference

**Target:** semua model terhubung.

```text
ESP32 → YOLO → MediaPipe → CNN → LSTM → class
```

### Fase 7 — Sentence Builder + TTS

**Target:** gesture berubah menjadi kalimat sederhana dan suara.

### Fase 8 — Website + MongoDB

**Target:** seluruh fungsi manajemen tersedia melalui dashboard.

### Fase 9 — End-to-end testing

**Target:** sistem diuji menggunakan ESP32-CAM nyata pada berbagai kondisi.

### Fase 10 — Pengembangan tambahan

Setelah baseline stabil:

- SBC/Raspberry Pi;
- UART;
- cloud training;
- model optimization;
- distributed inference;
- deployment pipeline yang lebih matang.

---

## 40. Milestone yang Harus Dianggap “Selesai”

### Milestone A

```text
ESP32-CAM → Server
```

Syarat selesai: frame dapat dikirim dan diterima stabil.

### Milestone B

```text
Server → YOLO
```

Syarat selesai: tangan dapat dilokalisasi.

### Milestone C

```text
YOLO → MediaPipe
```

Syarat selesai: ROI menghasilkan landmark secara konsisten pada data valid.

### Milestone D

```text
MediaPipe/ROI → CNN
```

Syarat selesai: CNN menghasilkan feature/prediksi yang dapat dievaluasi.

### Milestone E

```text
CNN → LSTM
```

Syarat selesai: sequence feature dapat diproses dan menghasilkan prediksi.

### Milestone F

```text
Gesture → Sentence → TTS
```

Syarat selesai: satu rangkaian subjek + predikat menghasilkan suara yang sesuai.

### Milestone G

```text
Full System
```

Syarat selesai: Developer dapat mengelola dataset/model dan User dapat menjalankan deteksi real-time.

---

## 41. Prioritas Ketika Waktu Terbatas

Karena proyek D3 harus selesai dalam waktu yang terbatas, prioritas pembangunan sebaiknya:

```text
P0 — Wajib
ESP32-CAM
YOLO hand detection
MediaPipe landmarks
CNN
LSTM
```text
P0 — Wajib
ESP32-CAM (akuisisi citra stabil)
YOLO hand detection (YOLOv8n)
MediaPipe hand landmarks (21 titik 3D)
1D-CNN (ekstraksi spasial landmark)
LSTM (klasifikasi temporal sekuens)
Sentence Builder (Subjek + Predikat)
TTS (konversi audio)
Website dasar (FastAPI)
SQLite dasar (database lokal tanpa daemon)
Testing utama

P1 — Penting
Model Management (Upload, Hot-Swap, Rollback)
Dataset Management
Live Testing
Training evaluation display
Prediction stabilization
API documentation

P2 — Pengembangan
SBC (Raspberry Pi 5 4GB)
UART
Cloud GPU automation
Advanced model versioning
Model optimization (INT8 Quantization)
Distributed deployment
```

Fitur P2 tidak boleh menghambat penyelesaian P0.

---

## 42. Risiko Teknis dan Mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| YOLO gagal mendeteksi tangan | pipeline berhenti | variasikan dataset YOLO + fallback handling |
| MediaPipe gagal pada ROI | feature tidak tersedia | validasi crop dan skip frame invalid |
| 1D-CNN overfitting | generalisasi buruk | augmentasi koordinat, dropout, validation |
| LSTM overfitting | prediksi sequence buruk | sequence diversity + early stopping |
| Dataset tidak seimbang | kelas tertentu buruk | distribusi sample diperiksa |
| Frame duplicate | kalimat berulang | stabilization/cooldown |
| Wi-Fi tidak stabil | latency/drop frame | reconnect + heartbeat + buffer terbatas |
| Training lambat | waktu pengerjaan panjang | pre-generate landmark dataset + Colab bila perlu |
| Model terlalu besar | deployment sulit | gunakan 1D-CNN ringan dan format ONNX/TFLite |
| Scope melebar | TA terlambat | P0/P1/P2 dikontrol ketat |

---

## 43. Kriteria Keberhasilan Akhir

Sistem dianggap berhasil secara fungsional apabila:

1. ESP32-CAM mampu mengirim citra secara real-time ke server.
2. YOLO mampu mendeteksi objek `hand` dan menghasilkan bounding box.
3. MediaPipe mampu menghasilkan 21 landmark tangan dari area tangan yang valid.
4. 1D-CNN mampu menghasilkan representasi/fitur spasial dari koordinat landmark.
5. LSTM mampu memproses sequence dan menghasilkan prediksi gesture.
6. Gesture dikenali sesuai 14 kelas dataset yang digunakan.
7. Temporary buffer dapat menyimpan gesture dalam urutan yang benar.
8. Subjek + predikat dapat membentuk kalimat sederhana.
9. Kalimat dapat dikonversi menjadi suara melalui TTS.
10. Developer dapat mengelola dataset, mengevaluasi training, dan mengelola model.
11. User dapat melakukan deteksi tanpa hak mengubah model/dataset.
12. Hasil pengujian model dan sistem dapat didokumentasikan.
13. Sistem tetap dapat menangani frame atau koneksi yang gagal tanpa menghentikan aplikasi secara keseluruhan.

---

## 44. Checklist Implementasi Praktis

### Perangkat

- [ ] ESP32-CAM siap
- [ ] kamera teruji
- [ ] Wi-Fi teruji
- [ ] server/laptop dapat menerima frame

### Dataset

- [ ] daftar gesture final sudah disepakati
- [ ] YOLO dataset tersedia
- [ ] gesture dataset tersedia
- [ ] landmark extraction pre-processing tersedia
- [ ] train/val/test dipisahkan
- [ ] data blur/noise dibersihkan
- [ ] metadata dataset dicatat

### ML

- [ ] YOLO hand detector berhasil
- [ ] MediaPipe landmarks berhasil
- [ ] normalisasi landmark konsisten
- [ ] 1D-CNN berhasil dilatih
- [ ] LSTM berhasil dilatih
- [ ] model diekspor ke ONNX / TFLite
- [ ] evaluasi tersedia

### Runtime

- [ ] YOLO → ROI
- [ ] ROI → MediaPipe
- [ ] MediaPipe (21 landmarks) → 1D-CNN
- [ ] 1D-CNN → sequence buffer
- [ ] sequence → LSTM
- [ ] LSTM → gesture
- [ ] gesture → sentence
- [ ] sentence → TTS

### Website

- [ ] Login
- [ ] Role Developer/User
- [ ] Dashboard Developer
- [ ] Dataset Management
- [ ] Model Management (Upload, Hot-Swap, Rollback)
- [ ] Live Testing
- [ ] Dashboard User
- [ ] Detection history (SQLite)

### Pengujian

- [ ] YOLO metrics
- [ ] classification metrics
- [ ] confusion matrix
- [ ] blackbox testing
- [ ] latency test
- [ ] connectivity test
- [ ] environmental test

---

## 45. Kontrak Arsitektur yang Perlu Dipertahankan

Agar pengembangan tidak kembali berubah-ubah, gunakan kontrak berikut sebagai acuan kerja:

```text
[ ESP32-CAM ]
      |
      | frame (MJPEG / Raw Binary)
      v
[ Communication Layer ]
      |
      v
[ YOLO: hand + bbox ]
      |
      v
[ ROI Tangan ]
      |
      v
[ MediaPipe: 21 landmarks (3D) ]
      |
      v
[ Normalisasi Koordinat (Wrist-centered) ]
      |
      v
[ 1D-CNN: spatial representation ]
      |
      v
[ LSTM: temporal representation ]
      |
      v
[ Gesture Class ]
      |
      v
[ Sentence Builder (S + P) ]
      |
      v
[ TTS Engine ]
```

Sedangkan website berada di sisi pengelolaan:

```text
                 ┌─────────────────────┐
                 │    Web Dashboard    │
                 │ Developer / User    │
                 └──────────┬──────────┘
                            │
                    ┌───────▼────────┐
                    │  FastAPI App   │
                    └───────┬────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
       ┌──────▼──────┐             ┌──────▼──────┐
       │  ML Engine  │             │   SQLite    │
       └──────┬──────┘             └─────────────┘
              │
       ┌──────▼──────┐
       │ Active Model│
       └─────────────┘
```

---

## 46. Hubungan dengan Proposal

Dokumen ini mempertahankan bagian utama proposal:

| Bagian proposal | Implementasi pada roadmap |
|---|---|
| ESP32-CAM sebagai perangkat akuisisi | Bagian 2, 18, 39 |
| Wi-Fi ke server | Bagian 2, 18, 39 |
| YOLO | Bagian 3, 7, 10, 28 |
| MediaPipe 21 landmarks | Bagian 3, 7, 14, 39 |
| CNN | Bagian 3, 8, 10 (dioptimalkan sebagai 1D-CNN landmark) |
| LSTM | Bagian 3, 9, 10 |
| Temporary buffer | Bagian 14, 16 |
| Pembentukan kalimat | Bagian 16 |
| Text-to-Speech | Bagian 17 |
| Website | Bagian 21–24 |
| Database | Bagian 21 (diwujudkan sebagai SQLite) |
| Developer/User | Bagian 21, 37, 38 |
| Model Management | Bagian 12, 25 |
| Pengujian akurasi | Bagian 28–30 |
| Blackbox Testing | Bagian 30 |
| Analisis kondisi lingkungan | Bagian 29 |

---

## 47. Bagian Tambahan di Luar Proposal yang Dirancang sebagai Pengembangan

Beberapa hal di bawah ini sengaja diposisikan sebagai tambahan teknis karena tidak semuanya tertulis secara eksplisit pada proposal:

- pemisahan dataset YOLO dan gesture recognition;
- normalisasi koordinat landmark MediaPipe;
- optimasi 1D-CNN untuk representasi spasial sendi tangan;
- prediction stabilization;
- dataset versioning;
- model registry (upload, hot-swap, rollback);
- active pipeline configuration;
- ekspor model ke format edge (ONNX / TFLite);
- logging detail setiap tahap inference;
- API contract (FastAPI);
- error handling per stage;
- cloud GPU untuk training mandiri;
- SBC (Raspberry Pi 5 4GB);
- UART alternatif;
- observability dan debugging metadata.

Tambahan tersebut berfungsi untuk membuat implementasi lebih rapi dan mudah dikembangkan tanpa menghilangkan komponen yang sudah ditetapkan proposal.

---

## 48. Aturan Perubahan Selama Pembangunan

Setiap perubahan arsitektur harus dicatat dengan tiga kategori:

### A. Tidak mengubah proposal

Contoh:

- memperbaiki struktur kode;
- menambah logging;
- menambah versioning;
- mengubah parameter training;
- menambah mekanisme stabilization.

### B. Pengembangan teknis yang masih sejalan

Contoh:

- SBC/Raspberry Pi;
- UART;
- cloud GPU;
- model registry;
- optimasi inference.

### C. Perubahan yang memerlukan sinkronisasi proposal

Contoh:

- mengganti ESP32-CAM sebagai perangkat akuisisi utama;
- menghapus YOLO/CNN/LSTM dari arsitektur yang dijadikan fokus utama;
- mengganti output TTS menjadi jenis output yang berbeda;
- mengubah jumlah atau definisi kelas gesture secara permanen;
- memindahkan seluruh ML ke perangkat embedded dengan arsitektur yang berbeda secara fundamental;
- mengubah peran Developer/User secara signifikan.

---

## 49. Rencana Pembangunan Setelah Proposal

Urutan kerja praktis yang disarankan:

```text
1. Finalisasi daftar gesture
        ↓
2. Siapkan struktur repository
        ↓
3. Uji ESP32-CAM sebagai camera source
        ↓
4. Siapkan dataset YOLO
        ↓
5. Training YOLO hand detector
        ↓
6. Uji YOLO realtime
        ↓
7. Tambahkan crop ROI
        ↓
8. Integrasikan MediaPipe
        ↓
9. Visualisasikan 21 landmark
        ↓
10. Kumpulkan dataset gesture
        ↓
11. Bangun preprocessing gesture
        ↓
12. Training CNN
        ↓
13. Ekstraksi CNN feature
        ↓
14. Susun sequence
        ↓
15. Training LSTM
        ↓
16. Integrasikan pipeline realtime
        ↓
17. Tambahkan stabilization
        ↓
18. Tambahkan Sentence Builder
        ↓
19. Integrasikan TTS
        ↓
20. Bangun MongoDB
        ↓
21. Bangun Dashboard Developer
        ↓
22. Bangun Dataset Management
        ↓
23. Bangun Training Model
        ↓
24. Bangun Model Management
        ↓
25. Bangun Live Testing
        ↓
26. Bangun Dashboard User
        ↓
27. End-to-end testing
        ↓
28. Analisis hasil
        ↓
29. Dokumentasi
        ↓
30. Pengembangan tambahan bila baseline sudah stabil
```

---

## 50. Prinsip Akhir Pengembangan

1. **Bangun per komponen, kemudian integrasikan.** Jangan langsung menggabungkan ESP32-CAM + YOLO + MediaPipe + CNN + LSTM + Website + MongoDB pada tahap pertama.
2. **YOLO tetap menjadi hand detector**, bukan classifier 14 gesture.
3. **MediaPipe digunakan untuk struktur/landmark tangan**, bukan menggantikan seluruh proses klasifikasi.
4. **CNN menangani representasi spasial**, sedangkan **LSTM menangani urutan temporal**.
5. **Sentence Builder hanya membentuk kalimat sederhana** sesuai kelas yang tersedia, bukan parser bahasa natural umum.
6. **Model dan dataset harus memiliki versi** agar perubahan dapat dilacak.
7. **Baseline harus selesai sebelum fitur tambahan** seperti SBC, UART, optimasi embedded, atau cloud automation ditambahkan.
8. **Setiap hasil evaluasi harus berasal dari pengujian aktual**, bukan angka contoh atau asumsi.
9. **Perubahan yang mengubah inti proposal harus disinkronkan dengan proposal**, sedangkan peningkatan teknis yang tidak mengubah inti sistem dapat dicatat sebagai pengembangan.

---

## 51. Status Dokumen Kerja

Gunakan status berikut selama pembangunan:

```text
[ ] Belum dikerjakan
[~] Sedang dikerjakan
[x] Selesai
[!] Bermasalah / perlu perbaikan
[-] Ditunda
```

Contoh:

```text
[x] YOLO environment
[x] MediaPipe environment
[~] Dataset gesture
[ ] CNN training
[ ] LSTM training
[ ] Sentence Builder
[ ] TTS
[ ] MongoDB
[ ] Web Dashboard
[ ] End-to-end testing
```

Dokumen ini sebaiknya diperbarui setiap kali terjadi perubahan besar pada arsitektur, dataset, model, atau metode deployment.
