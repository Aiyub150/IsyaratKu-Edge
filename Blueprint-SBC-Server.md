# Blueprint 2 — Deployment SBC dan Pemeliharaan Sistem Hand Gesture

## 1. Tujuan Blueprint

Blueprint ini merupakan pengembangan teknis dari proposal tugas akhir **“Implementasi Machine Learning untuk Deteksi Hand Gesture pada Sistem IoT Berbasis ESP32-CAM”**.

Proposal tetap menjadi dasar utama sistem. Blueprint ini menambahkan rancangan deployment agar sistem tidak berhenti pada tahap demonstrasi menggunakan laptop sebagai server, tetapi dapat ditempatkan di lingkungan masyarakat dengan **Single Board Computer (SBC)** sebagai **local edge server**.

Tujuan utama pengembangan ini adalah:

1. ESP32-CAM tetap digunakan sebagai perangkat akuisisi citra dan node IoT.
2. SBC menggantikan laptop sebagai server lokal yang berjalan terus-menerus.
3. Beban komputasi machine learning dipusatkan pada SBC sehingga ESP32-CAM tidak menjalankan model berat secara penuh.
4. Sistem dapat dipelihara dari perangkat lain melalui jaringan tanpa harus memasang monitor, keyboard, atau GUI langsung pada SBC.
5. Administrasi, pemeriksaan kesehatan sistem, pergantian model, pengelolaan konfigurasi, dan pemulihan layanan dibuat sesederhana mungkin.
6. Laptop tetap digunakan sebagai perangkat pengembangan, training, evaluasi, dan eksperimen sehingga SBC tidak dibebani pekerjaan training yang tidak diperlukan untuk operasi harian.

---

# 2. Prinsip Arsitektur

Arsitektur utama yang dipertahankan dari proposal:

```text
ESP32-CAM
   │
   │ Wi-Fi
   ▼
Server / Local Processing
   │
   ▼
Machine Learning
   │
   ▼
Teks → TTS
```

Dalam implementasi lapangan, istilah **server** diwujudkan menggunakan SBC (**Raspberry Pi 5 4GB**):

```text
                 JARINGAN LOKAL
             Wi-Fi (MJPEG / Stream)
                        │
                        │
              ┌─────────▼─────────┐
              │     ESP32-CAM     │
              │                   │
              │ OV2640 Camera     │
              │ Akuisisi Frame    │
              │ Resolusi VGA/QVGA │
              │ Stream Buffer     │
              └─────────┬─────────┘
                        │
                        │ Frame (Raw Binary / MJPEG)
                        ▼
              ┌─────────────────────────────────────┐
              │         SBC (Raspberry Pi 5)        │
              │          LOCAL EDGE SERVER          │
              │                                     │
              │  1. FastAPI Gateway / Stream Ingest │
              │  2. YOLOv8n (Presence Hand Bbox)    │
              │  3. MediaPipe Hands (21 3D Points)  │
              │  4. Landmark Normalizer (63 float)  │
              │  5. 1D-CNN Spatial Encoder          │
              │  6. LSTM Temporal Sequence          │
              │  7. Sentence Builder (S + P)        │
              │  8. Text-to-Speech (TTS Engine)     │
              │  9. Web Maintenance Dashboard       │
              │ 10. SQLite Database (JSON enabled)  │
              └─────────────────────────────────────┘
                        │
                        ▼
                Teks / Keluaran Suara
```

---

# 3. Pembagian Peran Perangkat

## 3.1 Laptop Development

Laptop digunakan untuk pekerjaan yang membutuhkan komputasi lebih besar:

- pemrograman dan debugging;
- pengumpulan dan anotasi dataset;
- pra-ekstraksi koordinat landmark MediaPipe;
- training model (YOLO, 1D-CNN, LSTM);
- evaluasi model dan analisis confusion matrix;
- ekspor model ke format edge (**ONNX / TFLite**);
- pengembangan website dan API;
- pengujian sebelum deployment ke SBC.

Laptop **tidak perlu hidup 24 jam** untuk membuat sistem lapangan berjalan.

## 3.2 ESP32-CAM

ESP32-CAM berfungsi murni sebagai node IoT dan perangkat akuisisi citra:

- inisialisasi modul kamera OV2640 pada resolusi optimal (VGA $640 \times 480$ atau QVGA $320 \times 240$);
- mengambil frame secara real-time (target 15–20 FPS);
- terhubung ke jaringan Wi-Fi lokal secara stabil;
- mengirimkan stream citra ke SBC menggunakan protokol minim overhead (MJPEG over HTTP atau WebSocket raw binary);
- menjaga kestabilan daya (konsumsi arus stabil tanpa thermal throttling).

> **Catatan Kritis:** Komputasi CNN pada ESP32-CAM **ditiadakan**. ESP32-CAM difokuskan 100% pada kestabilan video acquisition untuk mencegah *watchdog reset*, *brownout*, dan *memory contention*.

## 3.3 SBC (Raspberry Pi 5 4GB)

SBC menjadi pusat operasi lapangan dan berfungsi sebagai **local edge server**:

- menerima dan mendecoding stream frame dari ESP32-CAM;
- menjalankan pipeline machine learning yang telah dioptimasi:
  - YOLOv8n untuk deteksi kehadiran tangan (hand presence);
  - MediaPipe Hands untuk mengekstraksi 21 koordinat landmark 3D;
  - 1D-CNN untuk mengekstraksi korelasi spasial antar sendi;
  - LSTM untuk analisis dinamika temporal gesture;
- menjalankan Sentence Builder (State Machine: Subjek + Predikat);
- menjalankan engine TTS untuk output suara lokal;
- menjalankan backend **FastAPI** dan web dashboard pemeliharaan;
- mengelola konfigurasi, log, dan riwayat deteksi menggunakan **SQLite** (tanpa beban daemon server);
- mengelola model aktif (Model Registry, Hot-Swap, Rollback).

---

# 4. Pipeline Machine Learning pada SBC

Pipeline utama dirancang sangat ramping dan hemat daya:

```text
Frame dari ESP32-CAM (VGA/QVGA)
        │
        ▼
   YOLOv8n (ONNX)
        │
        │ Hand Presence Check + Bounding Box
        ▼
   Crop ROI Tangan (dengan margin ~15%)
        │
        ▼
  MediaPipe Hands
        │
        │ 21 Hand Landmarks (x, y, z)
        ▼
  Normalisasi Landmark
        │
        │ Vektor spasial 63 float (Wrist-centered)
        ▼
      1D-CNN
        │
        │ Spatial feature representation (64-dimensi)
        ▼
       LSTM
        │
        │ Temporal sequence (12 frame buffer)
        ▼
  Gesture Class (14 Kelas) + Confidence
        │
        ▼
  Stabilization Filter (Majority voting / threshold)
        │
        ▼
  Sentence Builder (Subjek + Predikat)
        │
        ▼
       TTS
        │
        ▼
  Keluaran Suara
```

### Pembagian fungsi

| Komponen | Fungsi | Lokasi utama | Format Runtime |
|---|---|---|---|
| ESP32-CAM Camera | Akuisisi citra & stream | ESP32-CAM | C++ / FreeRTOS |
| YOLOv8n | Deteksi objek `hand` & trigger | SBC | ONNX / TFLite (INT8/FP16) |
| MediaPipe Hands | Ekstraksi 21 landmark tangan 3D | SBC | Python / MediaPipe C++ |
| Normalizer | Normalisasi koordinat pergelangan | SBC | NumPy (<0.1 ms) |
| 1D-CNN | Ekstraksi korelasi spasial sendi | SBC | ONNX / TFLite (<1 ms) |
| LSTM | Pemrosesan urutan temporal | SBC | ONNX / TFLite (<2 ms) |
| Sentence Builder | Menggabungkan kata menjadi kalimat (S+P) | SBC | Python State Machine |
| TTS | Mengubah teks menjadi suara | SBC | Piper / eSpeak / gTTS |
| Database | Penyimpanan log, user, config | SBC | SQLite (in-process) |

Catatan penting: YOLO dengan satu kelas `hand` tidak digunakan untuk membedakan `Saya`, `Makan`, `Kami`, dan kelas gesture lainnya. Kelas gesture diproses pada tahap pengenalan gesture setelah area tangan diperoleh.

---

# 5. Dataset dan Training

Training tidak perlu dilakukan pada SBC setiap kali sistem digunakan.

Alur yang disarankan:

```text
Pengumpulan Dataset
        ↓
Quality Check
        ↓
Labeling
        ↓
Train / Validation / Test
        ↓
Training di Laptop / Cloud GPU
        ↓
Evaluation
        ↓
Model Release
        ↓
Deployment ke SBC
        ↓
Inference Real-time
```

## 5.1 Dataset YOLO

Dataset YOLO digunakan untuk mendeteksi objek tangan.

Kelas dasar:

```text
hand
```

Dataset harus mencakup variasi:

- posisi tangan;
- jarak dari kamera;
- sudut tangan;
- pencahayaan;
- latar belakang;
- jenis pose, termasuk pose yang mudah gagal dideteksi seperti tangan mengepal.

## 5.2 Dataset Gesture Recognition

Dataset gesture digunakan untuk membedakan kelas bahasa isyarat/gesture yang menjadi target tugas akhir.

Kelas yang digunakan harus mengikuti versi daftar gesture terbaru yang telah disepakati pada proposal. Proposal saat ini memuat 14 kelas awal:

### Subjek

```text
Saya
Kamu
Anda
Kami
Kita
Dia
Mereka
```

### Predikat

```text
Makan
Minum
Tidur
Belajar
Bekerja
Berjalan
Membaca
```

Apabila daftar tersebut berubah, seluruh dataset, model, website, dokumentasi, dan pengujian harus menggunakan daftar versi yang sama.

---

# 6. Model Registry dan Model Management

SBC tidak melakukan training dari nol untuk setiap model.

Model dilatih pada lingkungan development, kemudian dipindahkan ke SBC sebagai model yang siap digunakan.

Contoh:

```text
models/
├── yolo/
│   ├── hand-v1.pt
│   └── hand-v2.pt
│
├── cnn/
│   ├── gesture-v1.keras
│   └── gesture-v2.keras
│
└── lstm/
    ├── gesture-sequence-v1.keras
    └── gesture-sequence-v2.keras
```

Pada website, setiap model mempunyai metadata:

```text
Model Name
Version
Model Type
Dataset Version
Accuracy
Precision
Recall
F1-Score
Created At
Status
```

Status dapat dibuat sederhana:

```text
ACTIVE
INACTIVE
TESTING
ARCHIVED
```

Hanya model `ACTIVE` yang dipakai oleh pipeline produksi.

---

# 7. Website sebagai Pusat Pemeliharaan

Salah satu prinsip penting deployment adalah:

> **SBC tidak harus memiliki monitor dan GUI lokal untuk melakukan administrasi normal.**

Administrator cukup membuka browser dari laptop/PC/ponsel yang berada pada jaringan yang sama.

Contoh:

```text
http://192.168.1.50:8080
```

atau:

```text
http://192.168.1.50:8080/admin
```

Alamat dan port tersebut hanya contoh; nilai sebenarnya ditentukan saat deployment.

---

# 8. Dashboard Operasional / Maintenance

Selain dashboard Developer dan User yang telah dirancang dalam proposal, versi deployment sebaiknya memiliki bagian **System Maintenance**.

## 8.1 System Status

Menampilkan:

```text
SBC Status           : ONLINE (Raspberry Pi 5)
Uptime               : 17 days
CPU                  : 12%
RAM                  : 24% (dari 4 GB)
Storage              : 35%
Temperature          : 46°C (Active Cooler Normal)
Database (SQLite)    : ONLINE (In-Process, 0 MB overhead)
ML Engine            : ONLINE (ONNX Runtime)
Web Service          : ONLINE (FastAPI)
ESP32-CAM            : CONNECTED (18 FPS)
Active Model         : gesture-v2 (1D-CNN + LSTM)
```

Tujuannya agar masalah umum dapat diketahui tanpa masuk ke terminal SBC.

## 8.2 Service Control

Administrator dapat melihat dan menjalankan tindakan terbatas:

```text
ML Engine      [Restart]
Web Server     [Restart]
Database       [Vacuum / Backup]
ESP32 Gateway  [Reconnect Stream]
```

Tombol restart hanya boleh melakukan restart service yang diperlukan, bukan mematikan seluruh SBC.

## 8.3 Log Viewer

Administrator dapat membaca log melalui browser:

```text
INFO  ESP32 connected
INFO  Frame received
INFO  YOLO detected hand
INFO  MediaPipe landmarks extracted
WARN  LSTM confidence below threshold
ERROR TTS service unavailable
```

Log tidak perlu menampilkan seluruh isi log sistem operasi. Website hanya menampilkan log aplikasi yang relevan.

---

# 9. Remote Maintenance tanpa GUI SBC

Target operasionalnya:

```text
Operator
   │
   │ Browser
   ▼
http://SBC-IP:PORT
   │
   ▼
Admin Dashboard
   ├── Health Check
   ├── Device Status
   ├── Model Management
   ├── Configuration
   ├── Logs
   ├── Backup
   └── Service Control
```

Dengan desain tersebut, operator normal tidak perlu:

```text
Monitor
Keyboard
Mouse
Desktop GUI
```

yang terhubung langsung ke SBC.

SSH tetap dapat disediakan sebagai **jalur darurat untuk developer**, tetapi bukan jalur utama pemeliharaan harian.

---

# 10. Fitur Maintenance yang Disarankan

## 10.1 Restart ML Service

Ketika pipeline ML macet:

```text
Admin Dashboard
      ↓
Restart ML Engine
      ↓
Service berhenti
      ↓
Service dijalankan kembali
      ↓
Health Check
      ↓
ONLINE
```

Tidak perlu reboot SBC seluruhnya apabila hanya ML Engine yang bermasalah.

## 10.2 Restart Web Service

Jika dashboard tidak dapat diakses tetapi SBC tetap hidup, administrator dapat melakukan restart web service.

## 10.3 Reconnect ESP32-CAM

Website dapat menyediakan tindakan untuk meminta gateway melakukan reconnect terhadap perangkat ESP32-CAM.

## 10.4 Reload Model

Setelah model baru ditempatkan di SBC:

```text
Upload model
      ↓
Validate model
      ↓
Register model
      ↓
Set model ACTIVE
      ↓
Reload ML Engine
      ↓
Health Check
```

---

# 11. Update Model Tanpa Masuk ke SBC

Salah satu fitur maintenance yang paling berguna adalah **deployment model melalui website**.

Alurnya:

```text
Laptop Developer
      ↓
Upload Model
      ↓
SBC Admin Dashboard
      ↓
Validation
      ↓
Model Registry
      ↓
Test Load
      ↓
Deploy
      ↓
Model ACTIVE
```

Contoh:

```text
gesture-v1   ACTIVE
gesture-v2   TESTING
```

Setelah `gesture-v2` lolos validasi:

```text
gesture-v1   INACTIVE
gesture-v2   ACTIVE
```

Model lama **tidak langsung dihapus**, sehingga rollback dapat dilakukan apabila model baru bermasalah.

---

# 12. Rollback Model

Setiap deployment model sebaiknya dapat dibatalkan.

Contoh:

```text
ACTIVE MODEL
     │
     ▼
gesture-v3
     │
     │ masalah
     ▼
Rollback
     │
     ▼
gesture-v2
```

Rollback harus menjadi tindakan administrasi sederhana melalui dashboard.

---

# 13. Health Check API

Website menggunakan endpoint internal untuk memeriksa kondisi sistem secara periodik.

Contoh:

```text
GET /api/health
```

Respons:

```json
{
  "status": "ok",
  "device_esp32": "connected",
  "fps": 18.2,
  "ml_engine": "running",
  "database": "sqlite_ok",
  "active_model": "gesture-v2",
  "cpu_temp": "46.2C",
  "tts": "available"
}
```

Endpoint lain yang disediakan:

```text
GET  /api/device/status
GET  /api/model/active
POST /api/model/upload
POST /api/model/deploy
POST /api/model/rollback
POST /api/service/restart
GET  /api/logs/recent
GET  /api/system/resources
```

---

# 14. Auto Recovery dengan Systemd

Agar sistem lapangan tidak mudah berhenti karena satu modul mengalami error, seluruh aplikasi dijalankan di bawah pengelolaan **systemd**.

Konsep Auto-Recovery:

```text
ML Engine / Web Service Exception
       ↓
Process Terhenti
       ↓
Systemd mendeteksi (Restart=always, RestartSec=3)
       ↓
Restart otomatis dalam 3 detik
       ↓
Health Check pulih
       ↓
ONLINE
```

Urutan Boot Otomatis SBC:

```text
Power ON
   ↓
Raspberry Pi OS (64-bit Bookworm)
   ↓
Systemd Service Manager
   ↓
FastAPI Service (`gesture-edge.service`)
   ├── Inisialisasi SQLite (In-Process)
   ├── Load Active Model (ONNX Runtime)
   └── Buka Port Web & Stream Ingest (8000)
   ↓
ESP32-CAM Terhubung & Streaming
   ↓
System Ready (Operasional Penuh)
```

Contoh unit file `systemd` (`/etc/systemd/system/gesture-edge.service`):

```ini
[Unit]
Description=Hand Gesture Edge Server
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/hand-gesture-iot
ExecStart=/home/pi/hand-gesture-iot/venv/bin/uvicorn web.app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

---

# 15. Penyimpanan dan Backup (SQLite)

Karena menggunakan SQLite, manajemen penyimpanan dan backup menjadi sangat sederhana dan aman dari kerusakan memori MicroSD.

Data yang dicadangkan:

```text
1. database.sqlite3 (Konfigurasi, User, Log, Riwayat)
2. models/ (Artefak model ONNX & TFLite)
3. config/ (File konfigurasi sistem)
```

Skema Backup:

```text
SBC (Raspberry Pi 5)
 │
 ├── database.sqlite3
 │     │
 │     └── SQLite Online Backup API (VACUUM INTO)
 │           ↓
 └── Backup Tujuan
       ├── USB Flashdisk lokal (otomatis saat ditancapkan)
       └── Download via Web Admin Dashboard (1-Click Backup)
```

Keuntungan:
- Tidak memerlukan *database dump* rumit seperti `mongodump`.
- SQLite aman di-backup secara online tanpa menghentikan inferensi melalui perintah `VACUUM INTO 'backup.sqlite3'`.

---

# 16. Recovery dan Troubleshooting

## Level 1 — Masalah Aplikasi

Contoh: gesture tidak muncul, web masih aktif.

Tindakan:
- Buka Web Dashboard.
- Cek status stream ESP32-CAM dan FPS.
- Cek active model di Model Registry.
- Klik **Restart ML Service** pada menu Service Control.

## Level 2 — Masalah Koneksi ESP32-CAM

Contoh: ESP32 berstatus *Disconnected*.

Tindakan:
- Periksa lampu LED indikator daya ESP32-CAM.
- Klik tombol **Reconnect Stream** pada dashboard.
- Pastikan jarak ESP32-CAM ke Access Point Wi-Fi dalam jangkauan sinyal stabil.

## Level 3 — Masalah Service SBC

Contoh: Web dashboard tidak dapat diakses.

Tindakan:
- Cek IP SBC via router / ping.
- Systemd secara otomatis mencoba me-restart service.
- Jika tetap gagal, teknisi masuk via SSH.

## Level 4 — Jalur Darurat Teknisi (SSH)

Hanya digunakan saat pemeliharaan web tidak berfungsi:

```text
ssh pi@192.168.1.50
sudo systemctl status gesture-edge.service
sudo journalctl -u gesture-edge.service -n 50 --no-pager
```

---

# 17. Keamanan Remote Maintenance

Hak akses dibagi dua level:

| Fitur | User | Developer/Admin |
|---|---:|---:|
| Deteksi real-time & TTS | ✓ | ✓ |
| Melihat teks kalimat | ✓ | ✓ |
| Reset kalimat | ✓ | ✓ |
| Mengunggah model baru | ✗ | ✓ |
| Mengaktifkan / Rollback model | ✗ | ✓ |
| Restart service | ✗ | ✓ |
| Melihat log sistem | ✗ | ✓ |
| Download backup database | ✗ | ✓ |
| Konfigurasi perangkat | ✗ | ✓ |

Akses pemeliharaan dibatasi hanya pada jaringan Wi-Fi lokal atau VPN privat. Port sistem tidak dipublikasikan langsung ke internet publik tanpa proteksi.

---

# 18. Port dan Arsitektur Jaringan Terpadu

FastAPI menyatukan Web Dashboard, WebSocket frame stream, dan REST API ke dalam **satu port tunggal**:

```text
SBC IP : 192.168.1.50

Port 8000 : FastAPI Unified Server
            ├── Web Dashboard UI (/)
            ├── Admin Panel (/admin)
            ├── Stream Ingest WebSocket (/ws/stream)
            ├── Health Check API (/api/health)
            └── Model Management API (/api/model/*)

Port 22   : SSH (Jalur darurat teknisi)

SQLite    : In-Process (Tidak menggunakan port jaringan)
```

Manfaat arsitektur port tunggal:
- Tidak ada konflik port antar-service.
- Lebih hemat resource dan mudah diatur pada firewall.

---

# 19. Standarisasi Service Native (Penghapusan Docker)

Berdasarkan pertimbangan resource SBC (Raspberry Pi 5 RAM 4GB):

> **Pendekatan Container (Docker) ditiadakan; Sistem distandarisasi menggunakan Native Linux Service (Systemd).**

### Alasan Kritis Penghapusan Docker:

1. **Penghematan RAM:** *Docker daemon* (`dockerd` + `containerd`) memakan 250–350 MB RAM saat idle. Dengan native service, overhead ini menjadi **0 MB**.
2. **Kapasitas Penyimpanan MicroSD:** Image Docker untuk AI/ML (PyTorch, OpenCV, dependencies) berukuran 5–8 GB. Native Python virtual environment hanya memakan ~600 MB.
3. **Efisiensi I/O & Latensi:** Native service membaca stream jaringan dan hardware kamera secara langsung tanpa lapisan virtualisa# 21. Rasionalisasi dan Penghapusan CNN Ringan pada ESP32-CAM

Rencana awal menyertakan CNN ringan pada ESP32-CAM sebagai pre-filter frame. Setelah evaluasi teknis mendalam berbasis data hardware, **fitur ini resmi ditiadakan**.

### Alasan Kritis Penghapusan:

1. **Keterbatasan Hardware & Memory Contention:**
   ESP32-CAM ditenagai oleh prosesor Xtensa LX6 dual-core @ 240 MHz dengan SRAM internal 520 KB dan PSRAM 4 MB. Menjalankan TFLite for Microcontrollers (TFLM) membutuhkan waktu inferensi 150–300 ms per frame, yang langsung memangkas frame rate menjadi hanya **3–4 FPS**.
2. **Risiko Watchdog Timer (WDT) Reset & Brownout:**
   Kamera OV2640 dan modul Wi-Fi bekerja menggunakan DMA (*Direct Memory Access*). Ketika CPU terbebani inferensi CNN, *Wi-Fi transmission buffer* mengalami *underrun/overflow*, memicu WDT Reset (ESP32 me-restart tiba-tiba). Selain itu, beban simultan Wi-Fi + CNN memicu lonjakan arus >450 mA yang kerap menyebabkan *brownout* jika adaptor daya tidak sempurna.
3. **Peningkatan Panas Ekstrem:**
   Beban kerja komputasi tinggi pada ESP32-CAM menaikkan temperatur chip di atas 65°C, yang mempercepat penurunan kualitas sensor gambar (*thermal noise*).

### Keputusan Final:
ESP32-CAM difokuskan **100% sebagai Node Akuisisi Citra dan Video Streaming (MJPEG/WebSockets)** pada resolusi VGA/QVGA. Seluruh komputasi visual dipindahkan ke Raspberry Pi 5 yang memiliki daya komputasi ribuan kali lebih besar.

---

# 22. Operasi Normal di Lapangan

Penggunaan masyarakat dirancang semudah mungkin:

```text
Power ON
   ↓
SBC (Raspberry Pi 5) boot otomatis (~25 detik)
   ↓
ESP32-CAM boot otomatis & terhubung ke Wi-Fi (~5 detik)
   ↓
FastAPI Service aktif (Systemd auto-start)
   ↓
ML Engine & Stream Ingest aktif
   ↓
User membuka sistem di browser / display
   ↓
Deteksi Real-Time & Audio TTS Berjalan
```

Pengguna biasa tidak perlu membuka terminal Linux atau menjalankan skrip manual.

---

# 23. Operasi Developer/Admin

Developer/Admin mengakses halaman pemeliharaan melalui browser:

```text
http://192.168.1.50:8000/admin
```

Menu yang disediakan:

1. **System Status:** Pemantauan real-time CPU, RAM, Suhu, FPS, dan Status Koneksi ESP32-CAM.
2. **Model Registry:** Mengunggah model baru (`.onnx` / `.tflite`), melihat metadata, melakukan **Hot-Swap Deployment**, atau **Rollback**.
3. **Live Testing & Landmark Visualizer:** Menampilkan overlay 21 titik sendi tangan dan bounding box secara real-time.
4. **Service Control:** Tombol restart terisolasi untuk ML Engine, Web Server, atau Reconnect Stream.
5. **Log Viewer:** Membaca log operasional terkini dari sistem.
6. **Backup:** Mengunduh cadangan database `database.sqlite3` dengan satu kali klik.

> **Catatan:** Modul eksekusi training dihilangkan dari dashboard SBC. Training 100% dilakukan di Laptop/Cloud GPU, dan hasilnya diunggah ke SBC melalui menu Model Registry.

---

# 24. Alur Maintenance Saat Sistem Bermasalah

```text
Laporan Pengguna (Misal: Gesture Tidak Terdeteksi)
        ↓
Admin buka Web Dashboard (/admin)
        ↓
Periksa FPS Stream ESP32-CAM ──(0 FPS)──▶ Klik [Reconnect Stream] / Cek Fisik ESP32
        │
      (Normal)
        ↓
Periksa Active Model & ML Engine ──(Error)──▶ Klik [Restart ML Engine]
        │
      (Normal)
        ↓
Cek Log Aplikasi ──(Confidence Drop)──▶ Lakukan [Rollback Model] ke versi stabil
```

Jalur terminal SSH hanya dibuka jika web dashboard sama sekali tidak merespons.

---

# 25. Monitoring Resource SBC (Raspberry Pi 5)

Kondisi operasional normal pada Raspberry Pi 5 4GB:

```text
Perangkat      : Raspberry Pi 5 Model B (4 GB LPDDR4X)
CPU Usage      : 10% – 20% (Sangat Ringan berkat 1D-CNN)
RAM Usage      : ~800 MB (Termasuk OS, FastAPI, ONNX Runtime)
Storage        : ~4.5 GB (Termasuk OS Bookworm 64-bit)
Temperature    : 44°C – 48°C (Dengan Official Active Cooler)
Stream Input   : 18 – 20 FPS (VGA 640x480)
Inference Time : ~35 ms per frame (Total End-to-End)
Database       : SQLite (In-Process, 0 MB daemon)
```

---

# 26. Perawatan Fisik

Perhatian khusus untuk deployment lapangan:

- Gunakan **Power Supply resmi Raspberry Pi 27W USB-C PD** untuk menjamin kestabilan arus.
- Pasang **Official Active Cooler** (heatsink + kipas PWM) pada Raspberry Pi 5.
- Berikan jarak antena ESP32-CAM agar tidak terhalang logam.
- Bersihkan lensa kamera OV2640 secara periodik dari debu.

---

# 27. Panduan Pengguna Non-Teknis

## Sistem tidak mendeteksi gesture:
1. Pastikan lampu indikator pada ESP32-CAM menyala.
2. Pastikan tangan berada di area pandang kamera dengan pencahayaan memadai.
3. Buka halaman dashboard, periksa apakah indikator status berwarna hijau (`ONLINE`).

## Suara TTS tidak keluar:
1. Periksa kabel jack audio / speaker yang terhubung ke SBC.
2. Klik tombol `Speak` manual di dashboard untuk menguji keluaran audio.

---

# 28. Label Fisik pada Perangkat

Setiap node server lapangan ditempeli label:

```text
HAND GESTURE EDGE SERVER (RPi 5)
IP  : 192.168.1.50:8000
NODE: HGE-001 (Raspberry Pi 5 4GB)
```

---

# 29. Tahapan Implementasi yang Disarankan

1. **Tahap 1 — Pipeline ML di Laptop:** Training YOLOv8n, ekstraksi MediaPipe, training 1D-CNN + LSTM, Sentence Builder, dan TTS.
2. **Tahap 2 — Integrasi Streaming ESP32-CAM:** Menguji kestabilan stream VGA/QVGA via Wi-Fi ke laptop.
3. **Tahap 3 — Pindah Server ke Raspberry Pi 5:** Instalasi Raspberry Pi OS 64-bit, ekspor model ke format ONNX/TFLite, dan menjalankan inferensi edge.
4. **Tahap 4 — Web Dashboard + SQLite di SBC:** Mengintegrasikan FastAPI dan SQLite di Raspberry Pi 5.
5. **Tahap 5 — Remote Maintenance & Auto-Recovery:** Mengonfigurasi service `systemd` dan menu Model Registry.
6. **Tahap 6 — Optimasi Akhir:** Tuning threshold majority voting, stabilisasi kalimat, dan pengujian latensi.

---

# 30. Urutan Prioritas Fitur

## Wajib untuk sistem inti (P0):
- ESP32-CAM (Akuisisi citra)
- SBC Raspberry Pi 5 4GB
- YOLOv8n hand detector (ONNX)
- MediaPipe Hands (21 3D Landmarks)
- 1D-CNN (Spatial Landmark Encoder)
- LSTM (Temporal Classifier)
- Sentence Builder (Subjek + Predikat)
- Text-to-Speech (TTS)
- FastAPI Web Dashboard
- Database SQLite
- Role Developer & User

## Sangat disarankan untuk deployment nyata (P1):
- Auto-start & auto-restart via Systemd
- Health Check API
- Model Registry (Upload, Hot-Swap, Rollback)
- Log Viewer & 1-Click SQLite Backup

---

# 31. Arsitektur Final yang Direkomendasikan

```text
                         LAPTOP / CLOUD GPU
                     Development & Training
                               │
                               │ Model (.onnx / .tflite)
                               ▼
┌────────────────────────────────────────────────────────────┐
│                    SBC: RASPBERRY PI 5                     │
│                     LOCAL EDGE SERVER                      │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │             FastAPI Unified Server (Port 8000)       │  │
│  │  ├── Web Dashboard & Admin Maintenance Panel         │  │
│  │  ├── Stream Ingest WebSocket                         │  │
│  │  └── Model Registry & Health Check API               │  │
│  └──────────────────────────┬───────────────────────────┘  │
│                             │                              │
│                             ▼                              │
│                       ML ENGINE                            │
│                 YOLOv8n (Hand Detection)                   │
│                             │ ROI Crop                     │
│                             ▼                              │
│                 MediaPipe Hands (21 Landmarks)             │
│                             │ 63 float (Normalisasi)       │
│                             ▼                              │
│                 1D-CNN (Spatial Features)                  │
│                             │ 64-d vector                  │
│                             ▼                              │
│                 LSTM (Temporal Sequence)                   │
│                             │                              │
│                             ▼                              │
│                 Sentence Builder (S + P)                   │
│                             │                              │
│                             ▼                              │
│                    TTS Engine (Audio)                      │
│                             │                              │
│              ┌──────────────┴──────────────┐               │
│              │      SQLite Database        │               │
│              │   In-Process (hand_gesture) │               │
│              └─────────────────────────────┘               │
└─────────────────────────────┬──────────────────────────────┘
                              │
                    Wi-Fi (MJPEG / Stream)
                              │
                              ▼
                    ┌──────────────────┐
                    │    ESP32-CAM     │
                    │  (Node Akuisisi) │
                    │                  │
                    │ OV2640 Sensor    │
                    │ Resolusi VGA     │
                    │ Stream Buffer    │
                    └──────────────────┘
```

---

# 32. Prinsip Utama Pembangunan

> **Sistem dibangun dengan prinsip kesederhanaan, keandalan, dan efisiensi komputasi.** Menghilangkan redundansi data dan proses komputasi yang tidak perlu membuat sistem dapat berjalan stabil 24/7 di lapangan tanpa risiko overheat atau kehabisan memori.

---

# 33. Hubungan dengan Proposal Tugas Akhir

Semua komponen inti proposal tetap dipertahankan 100%:

| Komponen Proposal | Realisasi pada Edge Deployment | Status |
|---|---|---|
| ESP32-CAM | Node akuisisi citra & video streaming | Dipertahankan |
| Wi-Fi | Media transmisi stream frame | Dipertahankan |
| Server | Diwujudkan sebagai SBC (Raspberry Pi 5 4GB) | Dipertahankan & Ditingkatkan |
| YOLO | YOLOv8n (Hand presence detector) | Dipertahankan & Dioptimasi |
| MediaPipe | MediaPipe Hands (21 3D landmarks) | Dipertahankan |
| CNN | 1D-CNN Spatial Landmark Encoder | Dipertahankan & Dioptimasi |
| LSTM | LSTM Temporal Sequence Classifier | Dipertahankan |
| Sentence Builder | State Machine Subjek + Predikat | Dipertahankan |
| TTS | Local TTS Engine | Dipertahankan |
| Website | FastAPI Unified Dashboard | Dipertahankan |
| Database | SQLite (In-Process, JSON enabled) | Dipertahankan (Alternatif Efisien) |
| Developer / User | Dua role hak akses | Dipertahankan |

---

# 34. Ketetapan Spesifikasi Final

Ketetapan teknis yang telah difinalisasi:

1. **SBC Resmi:** **Raspberry Pi 5 (RAM 4GB)** + Official Active Cooler + PSU 27W USB-C PD.
2. **Sistem Operasi:** Raspberry Pi OS 64-bit (Debian Bookworm).
3. **Backend Framework:** **FastAPI** (Python 3.11/3.12) berjalan di atas Uvicorn.
4. **Database:** **SQLite** (`database.sqlite3`) dengan mode WAL (*Write-Ahead Logging*).
5. **Format Model Inferensi:** **ONNX Runtime** (untuk YOLOv8n dan 1D-CNN) serta **TFLite** (untuk LSTM).
6. **Protokol Kamera:** **MJPEG over HTTP** atau **WebSocket raw binary**.
7. **Manajemen Service:** **Systemd** (`gesture-edge.service`).
8. **Port Jaringan:** Port tunggal **8000** untuk seluruh layanan web, stream, dan API.

---

# 35. Target Akhir

Target akhir sistem adalah:

```text
                 USER
                  │
                  ▼
         ┌─────────────────┐
         │  Hand Gesture   │
         │     System      │
         └────────┬────────┘
                  │
           ESP32-CAM Camera
                  │
                  ▼
               Wi-Fi
                  │
                  ▼
        ┌─────────────────────┐
        │ SBC: Raspberry Pi 5 │
        │   Local Edge Server │
        └──────────┬──────────┘
                   │
             ML Pipeline
                   │
       YOLOv8n → MediaPipe Hands
                   │ 21 Landmarks (63-d)
                   ▼
                 1D-CNN
                   │ 64-d vector
                   ▼
                  LSTM
                   │
          Sentence Builder (S + P)
                   │
                  TTS
                   │
                   ▼
             Audio Output

Administrator:
Browser → IP-SBC:8000/admin → Maintenance Dashboard
```

Dengan target tersebut, masyarakat menggunakan perangkat secara sederhana, sementara Developer/Admin memiliki jalur pemeliharaan melalui jaringan tanpa harus secara rutin masuk secara fisik ke SBC.

---

## Status Dokumen

**Blueprint:** 2 — Deployment SBC & Maintenance

**Status:** Rancangan teknis teroptimasi / development blueprint

**Hubungan dengan proposal:** Pengembangan dan perincian implementasi; tidak menggantikan komponen utama proposal.
