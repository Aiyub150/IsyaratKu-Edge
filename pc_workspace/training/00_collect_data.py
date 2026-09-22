"""
00_collect_data.py
Alat Pengambil Data Interaktif Kamera untuk 14 Kata SIBI/BISINDO IsyaratKu-Edge.

Fitur Unggulan:
- Navigasi kelas langsung di layar (Tekan 'N' untuk kelas berikutnya, 'P' untuk sebelumnya).
- Hitungan mundur visual otomatis (Countdown 3.. 2.. 1.. GO!) saat menekan SPACE.
- Perekaman sekuens 12 frame otomatis dengan progress bar.
- Panduan pose teks langsung di layar untuk masing-masing 14 kata.
- Indikator pencapaian kuota per kelas (Target: 30 sekuens).
"""

import os
import sys
import time
import json
import cv2
import numpy as np
from pathlib import Path

# Tambahkan root directory ke sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.mediapipe_extractor import MediaPipeExtractor
from core.landmark_normalizer import normalize_landmarks

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "labels.json")
OUTPUT_RAW_DIR = os.path.join(BASE_DIR, "data", "dataset_raw")
TARGET_SEQS_PER_CLASS = 30

# Panduan pose per kata untuk membantu perekam
GESTURE_TIPS = {
    "saya": "Telunjuk tunjuk ke arah dada (jarak 5-10cm), 4 jari lain mengepal.",
    "kamu": "Telunjuk lurus horizontal ke depan arah kamera, 4 jari lain mengepal.",
    "anda": "Kelima jari terbuka rapat dan lurus, telapak tangan sopan menghadap kamera.",
    "kami": "Tangan 'C' melengkung di depan dada, gerakkan mendekat ke dada sendiri.",
    "kita": "Tangan terbuka rileks, buat gerakan melingkar horizontal di depan dada.",
    "dia": "Telunjuk lurus menunjuk ke arah samping kanan/kiri, jari lain mengepal.",
    "mereka": "Telapak/telunjuk melakukan gerakan sapuan (sweep) melebar ke samping.",
    "makan": "Kelima ujung jari menguncup bersamaan di depan mulut berulang.",
    "minum": "Jari melingkar cangkir, ibu jari tegak mengarah ke bibir.",
    "tidur": "Telapak tangan terbuka datar ditempel di samping pipi/telinga (bantal).",
    "belajar": "Satu telapak tangan terbuka mendatar menghadap atas di dada (lembaran buku).",
    "bekerja": "Tangan mengepal erat melakukan gerakan mengetuk ke bawah berulang.",
    "berjalan": "Jari telunjuk & tengah lurus ke bawah, melangkah maju-mundur bergantian.",
    "membaca": "Satu telapak datar, telunjuk menelusuri baris bacaan dari kiri ke kanan."
}

def load_classes():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["classes"]

def count_saved_sequences(class_name):
    save_dir = os.path.join(OUTPUT_RAW_DIR, class_name)
    if not os.path.exists(save_dir):
        return 0
    return len([f for f in os.listdir(save_dir) if f.endswith(".npy")])

def run_collector():
    classes = load_classes()
    current_class_idx = 0

    print("==================================================")
    print("   ISYARATKU-EDGE: GUIDED DATA AUTO-COLLECTOR     ")
    print("==================================================")
    print("Pintasan Keyboard:")
    print("  - [SPACE] : Mulai Countdown 3-2-1 & Rekam 1 Sekuens (12 Frame)")
    print("  - [N]     : Pindah ke Kelas Berikutnya")
    print("  - [P]     : Pindah ke Kelas Sebelumnya")
    print("  - [Q]     : Selesai dan Keluar\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Kamera webcam tidak dapat dibuka!")
        return

    extractor = MediaPipeExtractor(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5)

    # State mesin perekaman
    STATE_IDLE = "IDLE"
    STATE_COUNTDOWN = "COUNTDOWN"
    STATE_RECORDING = "RECORDING"
    
    state = STATE_IDLE
    countdown_start = 0.0
    countdown_duration = 3.0  # 3 detik countdown
    seq_frames = []

    # Gambar panduan visual gestur SIBI resmi (docs/panduan_subjek.jpg & docs/panduan_predikat.jpg)
    docs_dir = Path(__file__).resolve().parent.parent.parent / "docs"
    guide_subjek_img = cv2.imread(str(docs_dir / "panduan_subjek.jpg"))
    guide_predikat_img = cv2.imread(str(docs_dir / "panduan_predikat.jpg"))
    help_page = 0  # 0: Kamera, 1: Panduan Subjek, 2: Panduan Predikat

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()

        current_class = classes[current_class_idx]
        save_dir = os.path.join(OUTPUT_RAW_DIR, current_class)
        os.makedirs(save_dir, exist_ok=True)
        seq_count = count_saved_sequences(current_class)

        # Ekstraksi landmark dengan MediaPipe
        found, landmarks, frame = extractor.extract(frame, draw_debug=True)

        # ----------------------------------------------------
        # Logika State Machine Perekaman
        # ----------------------------------------------------
        if state == STATE_COUNTDOWN:
            elapsed = now - countdown_start
            remaining = countdown_duration - elapsed
            if remaining > 0:
                count_val = int(remaining) + 1
                # Visual countdown besar di tengah layar
                cv2.putText(frame, str(count_val), (w // 2 - 40, h // 2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 255), 8, cv2.LINE_AA)
                cv2.putText(frame, "SIAPKAN POSE TANGAN...", (w // 2 - 180, h // 2 + 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                state = STATE_RECORDING
                seq_frames = []

        elif state == STATE_RECORDING:
            if found:
                norm_lm = normalize_landmarks(landmarks)
                seq_frames.append(norm_lm)
            
            # Progress bar perekaman
            rec_progress = len(seq_frames) / 12.0
            bar_w = int(rec_progress * (w - 100))
            cv2.rectangle(frame, (50, h // 2 - 20), (50 + bar_w, h // 2 + 10), (0, 0, 255), -1)
            cv2.rectangle(frame, (50, h // 2 - 20), (w - 50, h // 2 + 10), (255, 255, 255), 2)
            cv2.putText(frame, f"MEREKAM SEKUENS... ({len(seq_frames)}/12)", (w // 2 - 140, h // 2 - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if len(seq_frames) >= 12:
                # Simpan sekuens ke file .npy
                timestamp_ms = int(time.time() * 1000)
                seq_path = os.path.join(save_dir, f"seq_{timestamp_ms}.npy")
                np.save(seq_path, np.array(seq_frames, dtype=np.float32))
                print(f"[REKAM] Sukses: {seq_path} ({seq_count + 1}/{TARGET_SEQS_PER_CLASS})")
                state = STATE_IDLE
                seq_frames = []

        # ----------------------------------------------------
        # RENDER UI & HUD PANEL
        # ----------------------------------------------------
        # 1. Header Atas: Info Kelas & Kuota
        overlay_top = frame.copy()
        cv2.rectangle(overlay_top, (0, 0), (w, 80), (20, 20, 20), -1)
        cv2.addWeighted(overlay_top, 0.8, frame, 0.2, 0, frame)

        class_title = f"[{current_class_idx + 1}/14] KELAS: {current_class.upper()}"
        cv2.putText(frame, class_title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Progress kuota
        quota_pct = min(1.0, seq_count / float(TARGET_SEQS_PER_CLASS))
        quota_color = (0, 255, 0) if seq_count >= TARGET_SEQS_PER_CLASS else (0, 165, 255)
        cv2.putText(frame, f"Terkumpul: {seq_count}/{TARGET_SEQS_PER_CLASS} Sekuens", (w - 280, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, quota_color, 2)

        # Bar kuota mini
        cv2.rectangle(frame, (w - 280, 42), (w - 30, 52), (60, 60, 60), -1)
        cv2.rectangle(frame, (w - 280, 42), (w - 280 + int(250 * quota_pct), 52), quota_color, -1)

        # 2. Bar Bawah: Tips Pose & Tombol Navigasi
        overlay_bot = frame.copy()
        cv2.rectangle(overlay_bot, (0, h - 70), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay_bot, 0.85, frame, 0.15, 0, frame)

        tip_text = GESTURE_TIPS.get(current_class, "")
        cv2.putText(frame, f"Panduan: {tip_text}", (20, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 200), 1)
        cv2.putText(frame, "[SPACE] Rekam (3s) | [N] Berikutnya | [P] Sebelumnya | [H] Gambar Panduan | [Q] Selesai", 
                    (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 180, 180), 1)

        # Tampilkan Window Preview (Kamera atau Gambar Panduan Visual SIBI)
        if help_page == 1 and guide_subjek_img is not None:
            display_frame = cv2.resize(guide_subjek_img, (w, h))
            overlay_h = display_frame.copy()
            cv2.rectangle(overlay_h, (0, h - 50), (w, h), (15, 15, 15), -1)
            cv2.addWeighted(overlay_h, 0.85, display_frame, 0.15, 0, display_frame)
            cv2.putText(display_frame, "[PANDUAN SUBJEK (1/2)] Tekan 'H' untuk Panduan Predikat | Tekan 'ESC' untuk Tutup",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
            cv2.imshow("IsyaratKu Data Collector [GUIDED]", display_frame)
        elif help_page == 2 and guide_predikat_img is not None:
            display_frame = cv2.resize(guide_predikat_img, (w, h))
            overlay_h = display_frame.copy()
            cv2.rectangle(overlay_h, (0, h - 50), (w, h), (15, 15, 15), -1)
            cv2.addWeighted(overlay_h, 0.85, display_frame, 0.15, 0, display_frame)
            cv2.putText(display_frame, "[PANDUAN PREDIKAT (2/2)] Tekan 'H' atau 'ESC' untuk Kembali ke Kamera",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
            cv2.imshow("IsyaratKu Data Collector [GUIDED]", display_frame)
        else:
            cv2.imshow("IsyaratKu Data Collector [GUIDED]", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            if help_page > 0:
                help_page = 0
            else:
                break
        elif key == ord('q'):
            break
        elif key == ord('h'):
            help_page = (help_page + 1) % 3
        elif key == 32:  # SPACE: Mulai Countdown
            if state == STATE_IDLE:
                state = STATE_COUNTDOWN
                countdown_start = time.time()
        elif key == ord('n'):  # N: Next class
            current_class_idx = (current_class_idx + 1) % len(classes)
            state = STATE_IDLE
            seq_frames = []
        elif key == ord('p'):  # P: Previous class
            current_class_idx = (current_class_idx - 1 + len(classes)) % len(classes)
            state = STATE_IDLE
            seq_frames = []

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()
    print("[INFO] Perekaman data selesai.")

if __name__ == "__main__":
    run_collector()
