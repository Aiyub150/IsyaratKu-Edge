"""
00_collect_data.py
Alat Pengambil Data Kamera untuk Pengguna IsyaratKu-Edge.
Merekam frame atau ekstraksi landmark secara interaktif dari webcam/ESP32-CAM
untuk setiap kelas dari 14 SIBI/BISINDO.
"""

import os
import sys
import time
import json
import cv2
import numpy as np

# Tambahkan root directory ke sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.mediapipe_extractor import MediaPipeExtractor
from core.landmark_normalizer import normalize_landmarks

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "labels.json")
OUTPUT_RAW_DIR = os.path.join(BASE_DIR, "data", "dataset_raw")

def load_classes():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["classes"]

def run_collector():
    classes = load_classes()
    print("==================================================")
    print("   ISYARATKU-EDGE: INTERACTIVE DATA COLLECTOR     ")
    print("==================================================")
    print("Daftar Kelas Tersedia:")
    for idx, c in enumerate(classes):
        print(f"  [{idx:2d}] {c}")
    print("--------------------------------------------------")

    choice = input("Pilih nomor indeks kelas yang ingin direkam (0-13): ").strip()
    try:
        class_idx = int(choice)
        target_class = classes[class_idx]
    except Exception:
        print("Pilihan tidak valid, default ke 'saya' (0)")
        target_class = "saya"

    save_dir = os.path.join(OUTPUT_RAW_DIR, target_class)
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n[INFO] Menyimpan data untuk kelas: '{target_class}' di {save_dir}")
    print("[KONTROL]")
    print("  - Tekan SPACE : Ambil 1 sampel frame")
    print("  - Tekan 'R'   : Mulai/Stop perekaman sekuens (12 frame otomatis)")
    print("  - Tekan 'Q'   : Selesai dan keluar\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak dapat dibuka!")
        return

    extractor = MediaPipeExtractor(min_detection_confidence=0.6, min_tracking_confidence=0.5)
    sample_count = len(os.listdir(save_dir))
    recording_seq = False
    seq_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        found, landmarks, display_frame = extractor.extract(display_frame, draw_debug=True)

        # Header status
        status_text = f"Kelas: {target_class} | Total Sampel: {sample_count}"
        cv2.putText(display_frame, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if recording_seq:
            cv2.putText(display_frame, f"REKORD SEKUEN ({len(seq_frames)}/12)...", (15, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if found:
                norm_lm = normalize_landmarks(landmarks)
                seq_frames.append(norm_lm)
                if len(seq_frames) >= 12:
                    # Simpan sekuens ke .npy
                    seq_filename = os.path.join(save_dir, f"seq_{int(time.time()*1000)}.npy")
                    np.save(seq_filename, np.array(seq_frames, dtype=np.float32))
                    sample_count += 1
                    recording_seq = False
                    seq_frames = []
                    print(f"Sekuens tersimpan: {seq_filename}")

        cv2.imshow("IsyaratKu Data Collector", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == 32:  # SPACE: simpan frame tunggal
            img_filename = os.path.join(save_dir, f"img_{int(time.time()*1000)}.jpg")
            cv2.imwrite(img_filename, frame)
            sample_count += 1
            print(f"Foto tersimpan: {img_filename}")
        elif key == ord('r'):
            recording_seq = not recording_seq
            seq_frames = []
            print(f"Perekaman sekuens: {'AKTIF' if recording_seq else 'NONAKTIF'}")

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()
    print("[INFO] Perekaman data selesai.")

if __name__ == "__main__":
    run_collector()
