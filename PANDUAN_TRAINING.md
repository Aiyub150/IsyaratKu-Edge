# Panduan Lengkap Perekaman Dataset & Pelatihan Model 14 Kata (IsyaratKu-Edge)

Dokumen ini adalah buku panduan teknis operasional untuk melakukan **perekaman data riil kamera (*real-world data collection*)** dan **pelatihan ulang model (*retraining*)** untuk seluruh 14 kata SIBI/BISINDO pada sistem **`IsyaratKu-Edge`**.

---

## 📑 Daftar Isi
1. [Mengapa Kata "Saya" Sempat Terdeteksi sebagai "Anda"?](#1-mengapa-kata-saya-sempat-terdeteksi-sebagai-anda)
2. [Fitur Baru Alat Perekam Data (00_collect_data.py)](#2-fitur-baru-alat-perekam-data-00_collect_datapy)
3. [Panduan Anatomi & Pose 14 Kata Saat Perekaman](#3-panduan-anatomi--pose-14-kata-saat-perekaman)
   - [A. Kategori Subjek (7 Kata)](#a-kategori-subjek-7-kata)
   - [B. Kategori Predikat (7 Kata)](#b-kategori-predikat-7-kata)
4. [Langkah Demi Langkah: Dari Perekaman hingga Model Siap Pakai](#4-langkah-demi-langkah-dari-perekaman-hingga-model-siap-pakai)
5. [Verifikasi Hasil Pelatihan](#5-verifikasi-hasil-pelatihan)

---

## 1. Mengapa Kata "Saya" Sempat Terdeteksi sebagai "Anda"?

Pada pengujian tahap 2, ketika pengguna menunjukkan gestur **"Saya"**, model terkadang mengidentifikasinya sebagai **"Anda"** atau **"Kamu"**. 

Akar masalah teknisnya adalah:
1. **Keterbatasan Dataset Sintetis:** Model sebelumnya dilatih menggunakan koordinat matematis sintetis. Pada tangan manusia asli, ketika jari telunjuk diarahkan ke dada, jari tengah/manis/kelingking tertekuk ke telapak tangan dengan sudut kemiringan tertentu. Data sintetis tidak memiliki variasi tekukan jari dan perspektif kedalaman (*z-axis*) kamera yang nyata.
2. **Perbedaan Fitur Spasial yang Tipis:**
   - **`saya`**: Telunjuk lurus mengarah ke dada, 4 jari lainnya **mengepal/menekuk rapat ke telapak**.
   - **`anda`**: **Kelima jari terbuka lurus rapat** (telapak tangan rata menghadap kamera).
   - **`kamu`**: Telunjuk **lurus ke depan** mengarah ke lensa kamera.
3. **Solusi Definitif:** Merekam **30 sekuens data riil dari kamera** untuk setiap kata menggunakan modul pengambil data yang telah dioptimasi. Data nyata ini akan mengajarkan model 1D-CNN + LSTM batas pemisah fitur (*decision boundary*) yang akurat antara kepalan 4 jari ("Saya") dan telapak 5 jari terbuka ("Anda").

---

## 2. Fitur Baru Alat Perekam Data (`00_collect_data.py`)

Modul [`pc_workspace/training/00_collect_data.py`](file:///c:/Users/aiyub/Desktop/ML%20Hand%20Gesture/pc_workspace/training/00_collect_data.py) kini telah dilengkapi antarmuka interaktif cerdas:

- **Hitungan Mundur Otomatis (Countdown 3.. 2.. 1.. GO!):** Cukup tekan <kbd>SPACE</kbd>, sistem memberi jeda 3 detik agar Anda dapat memposisikan tangan dengan sempurna, lalu otomatis merekam 1 sekuens (12 frame) dengan *progress bar* merah.
- **Navigasi Kelas di Layar:**
  - Tekan <kbd>N</kbd> untuk berpindah ke kelas berikutnya.
  - Tekan <kbd>P</kbd> untuk berpindah ke kelas sebelumnya.
- **Indikator Kuota Target:** Menampilkan jumlah sekuens yang sudah terkumpul (contoh: `[18/30 Sekuens] (60%)`).
- **Panduan Pose Teks:** Teks panduan pose ditampilkan langsung di bilah bawah layar kamera sesuai kelas aktif.

---

## 3. Panduan Anatomi & Pose 14 Kata Saat Perekaman

> [!IMPORTANT]
> **Pedoman Umum Saat Merekam:**
> - **Jarak Kamera:** Variasikan jarak pengambilan data: 10 sekuens pada jarak dekat ($\pm 50\text{ cm}$), 10 sekuens pada jarak sedang ($\pm 65\text{ cm}$), dan 10 sekuens pada jarak $\pm 80\text{ cm}$.
> - **Pencahayaan:** Pastikan pencahayaan terang dari arah depan. Hindari posisi membelakangi jendela terang (*backlight*).
> - **Variasi Sudut:** Ambil beberapa sekuens dengan posisi tegak lurus, dan beberapa sekuens dengan tangan sedikit miring ($\pm 15^\circ$).

---

### A. Kategori Subjek (7 Kata)

#### 1. `saya` (Target: 30 Sekuens - Mendukung 2 Variasi SIBI Resmi)
- **Bentuk Jari (2 Pilihan Sesuai Kamus SIBI):**
  1. *Variasi A:* Jari telunjuk lurus menunjuk ke arah dada sendiri, 4 jari lainnya mengepal rapat di depan dada ($5\text{–}10\text{ cm}$).
  2. *Variasi B (Kamus SIBI Resmi):* **Ibu jari dan kelingking terbuka (Y-sign / Shaka)**, sedangkan jari telunjuk, tengah, dan manis mengepal rapat ke telapak, dengan **punggung tangan menghadap ke depan/kamera**.
- **Kunci Pembeda:** Pastikan kontur jari terlihat jelas oleh kamera.

#### 2. `kamu` (Target: 30 Sekuens - Mendukung 2 Variasi SIBI Resmi)
- **Bentuk Jari (2 Pilihan Sesuai Kamus SIBI):**
  1. *Variasi A:* Jari telunjuk lurus horizontal menunjuk langsung ke arah lawan bicara / kamera. 4 jari lainnya mengepal rapat.
  2. *Variasi B:* **Telapak tangan terbuka 5 jari sopan** menghadap ke arah lawan bicara / kamera (dialihkan dari rekaman telapak sebelumnya).
- **Posisi:** Setinggi dada/bahu, mengarah lurus ke depan.

#### 3. `anda` (Target: 30 Sekuens - Kamus SIBI Resmi)
- **Bentuk Jari:** **Mengepal tangan** dengan bagian **alas pergelangan / bawah kepalan tangan menghadap ke arah depan/kamera** seolah-olah sedang menggenggam sesuatu.
- **Bukan Telapak Terbuka:** Sesuai Kamus SIBI, "Anda" bukan telapak tangan terbuka (telapak terbuka adalah "Kamu"), melainkan kepalan tangan dengan alas pergelangan tangan mengarah ke depan.
- **Kunci Pembeda:** Kepalan tangan tertutup rapat, alas pergelangan tangan menghadap kamera.

#### 4. `kami` (Target: 30 Sekuens - Pose Diam SIBI Resmi)
- **Bentuk Jari:** **Bentuk Huruf "K" / "V" (2 Jari Tegak)**.
  - Jari telunjuk dan jari tengah lurus tegak ke atas membentuk huruf V/K.
  - Jari manis dan kelingking mengepal rapat ke telapak.
  - Ibu jari melipat menahan jari manis di depan telapak tangan.
- **Posisi:** **Diam stabil di depan dada**. Tidak perlu digerakkan melingkar/mengayun agar tracking MediaPipe tetap presisi dan bebas blur.
- **Kunci Pembeda:** Hanya 2 jari (telunjuk + tengah) yang tegak lurus ke atas.

#### 5. `kita` (Target: 30 Sekuens - Pose Diam SIBI Resmi)
- **Bentuk Jari:** **Bentuk Huruf "W" (3 Jari Tegak)**.
  - Tiga jari: telunjuk, jari tengah, dan jari manis lurus tegak ke atas membentuk huruf 'W'.
  - Jari kelingking tertekuk ke telapak dan ditahan oleh ibu jari.
- **Posisi:** **Diam stabil di depan dada**. Tidak perlu digerakkan agar deteksi konsisten dan akurat.
- **Kunci Pembeda:** Tepat 3 jari (telunjuk, tengah, manis) yang tegak lurus ke atas.

#### 6. `dia` (Target: 30 Sekuens - Kamus SIBI Resmi)
- **Bentuk Jari:** Jari telunjuk lurus menunjuk ke arah **samping** (kanan atau kiri). **3 jari lainnya dan ibu jari mengepal rapat ke telapak**.
- **Posisi:** Diarahkan dengan tegas ke arah samping secara **statis** (orang ketiga tunggal, tanpa ayunan ke samping).
- **Kunci Pembeda:** Tepat 1 jari menunjuk ke samping secara stabil.

#### 7. `mereka` (Target: 30 Sekuens - Kamus SIBI Resmi)
- **Bentuk Jari:** Dua jari (telunjuk dan jari tengah) bersamaan menunjuk ke arah **samping** (orang ketiga jamak), jari manis dan kelingking mengepal rapat.
- **Posisi & Gerak:** Diarahkan ke samping dengan **gerakan ayunan/sapuan horizontal (*lateral sweep*) perlahan**.
- **Kunci Pembeda:** 2 jari menunjuk ke samping disertai gerakan menyapu horizontal (membedakannya secara tegas dari "Dia").

---

### B. Kategori Predikat (7 Kata)

#### 8. `makan` (Target: 30 Sekuens)
- **Bentuk Jari:** Kelima ujung jari (ibu jari, telunjuk, tengah, manis, kelingking) **menguncup bersamaan**.
- **Posisi:** Ujung jari yang menguncup diarahkan mendekati bagian depan mulut berulang kali (gerakan menyuap).

#### 9. `minum` (Target: 30 Sekuens - Kamus SIBI Resmi)
- **Bentuk Jari:** **Memegang cangkir/gelas (C-shape grip)**.
  - Jari telunjuk hingga kelingking melengkung membentuk dinding cangkir.
  - Ibu jari berhadapan membentuk sisi lain cangkir.
- **Posisi & Gerak:** Tangan yang membentuk pegangan gelas digerakkan mendekat ke bibir dengan sedikit mendongak (gerakan minum).

#### 10. `tidur` (Target: 30 Sekuens)
- **Bentuk Jari:** Telapak tangan terbuka rapat dan datar.
- **Posisi:** Ditempelkan di samping pipi/telinga dengan kepala sedikit dimiringkan ke arah tangan (posisi bantal).

#### 11. `belajar` (Target: 30 Sekuens)
- **Bentuk Jari:** **Satu telapak tangan mendatar terbuka** menghadap ke atas di depan dada (merepresentasikan lembaran buku).
- **Posisi:** Tangan terbuka datar stabil di depan dada.
- **Tips Khusus:** Menggunakan variasi satu tangan terbuka mendatar memastikan MediaPipe melacak tangan dengan stabil tanpa *jitter* atau kebingungan tangan kiri vs kanan.

#### 12. `bekerja` (Target: 30 Sekuens)
- **Bentuk Jari:** Tangan mengepal erat.
- **Posisi:** Gerakkan tangan mengetuk ke bawah berulang kali seperti sedang memegang alat kerja/palu.

#### 13. `berjalan` (Target: 30 Sekuens)
- **Bentuk Jari:** Jari telunjuk dan jari tengah diluruskan ke bawah (seperti 2 kaki orang), jari lainnya ditekuk.
- **Posisi:** Gerakkan kedua jari tersebut maju-mundur bergantian seolah-olah sedang melangkah.

#### 14. `membaca` (Target: 30 Sekuens)
- **Bentuk Jari:** Satu tangan terbuka mendatar (buku), jari telunjuk tangan lainnya menelusuri baris bacaan.
- **Posisi:** Jari telunjuk bergerak perlahan dari kiri ke kanan di atas telapak tangan datar tersebut.

---

### C. Gestur Kontrol Tambahan: `SALAH` (Reset Kalimat)
- **Fungsi:** Sesuai Kamus SIBI (Feedback #4 Poin 6), gestur "Salah" digunakan jika kata subjek maupun predikat yang terdeteksi salah. Gestur ini berfungsi sebagai **kontrol reset kalimat** yang telah dibentuk agar pengguna dapat mengulang kalimat berulang kali tanpa harus menekan keyboard.
- **Bentuk Gestur:** Jari telunjuk ditekuk melengkung (hook/huruf D) dengan jari lain mengepal (simbol SIBI "Salah") atau gerakan mengusap/menyapu horizontal telapak tangan di depan kamera.
- **Perlakuan Sistem:** BUKAN merupakan kelas kosakata ke-15 (kosakata tetap ketat 14 kata), melainkan memicu eksekusi `sentence_builder.reset_sentence()`.

---

## 4. Langkah Demi Langkah: Dari Perekaman hingga Model Siap Pakai

Ikuti 5 langkah mudah berikut:

### Langkah 1: Jalankan Alat Perekam Data
Buka terminal (pastikan venv aktif):
```powershell
python pc_workspace/training/00_collect_data.py
```
1. Layar kamera akan muncul menampilkan kelas pertama: **`SAYA`**.
2. Arahkan tangan sesuai panduan di layar.
3. Tekan <kbd>SPACE</kbd> $\rightarrow$ tunggu hitungan mundur `3.. 2.. 1..` $\rightarrow$ tahan pose saat bilah merah merekam 12 frame.
4. Ulangi tekan <kbd>SPACE</kbd> sebanyak **30 kali** untuk kelas tersebut (sambil sedikit memvariasikan jarak dan sudut tangan).
5. Tekan <kbd>N</kbd> untuk berpindah ke kelas berikutnya (**`KAMU`**, **`ANDA`**, dst.) hingga ke-14 kelas terisi.
6. Tekan <kbd>Q</kbd> jika sudah selesai.

---

### Langkah 2: Bangun & Gabungkan Dataset
Jalankan skrip penyusun dataset untuk mengagregasi seluruh file rekaman `.npy` dari `data/dataset_raw/`:
```powershell
python pc_workspace/training/01_prepare_dataset.py
```
- Skrip ini akan memuat seluruh data riil dari kamera, melengkapinya dengan augmentasi spasial, menerapkan normalisasi *rigid palm-base*, dan menyimpannya ke `data/landmarks_cache/dataset_landmarks.npz`.

---

### Langkah 3: Latih Model 1D-CNN (Spatial Landmark Encoder)
```powershell
python pc_workspace/training/02_train_1dcnn.py
```
- Melatih representasi spasial 21 titik sendi tangan.
- Model akan otomatis diekspor ke format ONNX: [`data/models/spatial_1dcnn.onnx`](file:///c:/Users/aiyub/Desktop/ML%20Hand%20Gesture/data/models/spatial_1dcnn.onnx).

---

### Langkah 4: Latih Model Temporal LSTM
```powershell
python pc_workspace/training/03_train_lstm.py
```
- Melatih pengenalan dinamika sekuens temporal 12 frame dari fitur 1D-CNN.
- Model akan otomatis diekspor ke format ONNX: [`data/models/temporal_lstm.onnx`](file:///c:/Users/aiyub/Desktop/ML%20Hand%20Gesture/data/models/temporal_lstm.onnx).

---

### Langkah 5: Uji Coba Langsung di Kamera
Jalankan modul pengujian pipeline lengkap:
```powershell
python pc_workspace/testing/test_pipeline_debug.py
```
- Praktikkan gestur **"Saya"** $\rightarrow$ perhatikan bahwa kata yang terkunci adalah **"Saya"** (bukan lagi "Anda").
- Praktikkan gestur **"Belajar"** $\rightarrow$ kalimat terangkai menjadi *"Saya belajar."*.
- Dengarkan keluaran suara TTS $\rightarrow$ audio penutur asli Bahasa Indonesia yang jernih dan natural langsung terdengar lewat speaker!

---

## 5. Verifikasi Hasil Pelatihan

| Skenario Pengujian | Hasil yang Diharapkan | Status Verifikasi |
| :--- | :--- | :---: |
| **Gestur "Saya"** | Kotak status mengunci kelas **`saya`** dengan keyakinan $\ge 80\%$. Tidak tertukar dengan `anda` atau `kamu`. | 🎯 Target Utama |
| **Gestur "Anda"** | Kotak status mengunci kelas **`anda`** saat 5 jari dibuka rapat ke arah kamera. | 🎯 Target Utama |
| **Gestur "Belajar"** | Kotak status mengunci kelas **`belajar`** saat telapak tangan dibuka mendatar di depan dada. | 🎯 Target Utama |
| **Penyusunan Kalimat** | *"Saya"* + *"Belajar"* $\rightarrow$ Kalimat *"Saya belajar."* tampil di layar selama 2.5 detik. | ✅ Terverifikasi |
| **Keluaran Suara TTS** | Memutar audio asli Bahasa Indonesia tanpa aksen asing (*`data/audio/saya_belajar.mp3`*). | ✅ Terverifikasi |
