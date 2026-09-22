import cv2
import os
import sys
import json
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from configs.settings import RAW_DATA_DIR, LANDMARKS_CACHE_DIR, LABELS_PATH
from core.mediapipe_extractor import MediaPipeExtractor
from core.landmark_normalizer import normalize_landmarks

def extract_dataset_landmarks():
    """
    SKRIP PRA-EKSTRAKSI DATASET LANDMARK:
    
    Membaca video dari folder `data/dataset_raw/<class_name>/*.mp4`,
    mengekstraksi 21 koordinat 3D MediaPipe per frame,
    menormalisasinya menjadi 63 float,
    dan menyimpannya ke `data/landmarks_cache/<class_name>.npy`.

    Manfaat Kritis:
    - Menghilangkan kebutuhan menjalankan MediaPipe berulang-ulang di setiap epoch training.
    - Mempercepat proses training hingga >20x lipat!
    """
    print("=" * 60)
    print("   ISYARATKU-EDGE: PRA-EKSTRAKSI DATASET LANDMARK   ")
    print("=" * 60)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels_data = json.load(f)
        classes = labels_data.get("classes", [])

    LANDMARKS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mp_extractor = MediaPipeExtractor(min_detection_confidence=0.6, min_tracking_confidence=0.5)

    total_classes = len(classes)
    for idx, class_name in enumerate(classes, 1):
        class_raw_dir = RAW_DATA_DIR / class_name
        output_file = LANDMARKS_CACHE_DIR / f"{class_name}.npy"

        print(f"\n[{idx}/{total_classes}] Memproses kelas: '{class_name}'...")
        if not class_raw_dir.exists():
            print(f"  [SKIP] Folder {class_raw_dir} belum memiliki data.")
            continue

        video_files = list(class_raw_dir.glob("*.mp4")) + list(class_raw_dir.glob("*.avi"))
        if not video_files:
            print(f"  [SKIP] Tidak ada file video di {class_raw_dir}")
            continue

        extracted_sequences = []
        for vid_path in video_files:
            cap = cv2.VideoCapture(str(vid_path))
            seq_landmarks = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                found, raw_lms, _ = mp_extractor.extract(frame, draw_debug=False)
                if found:
                    norm_vec = normalize_landmarks(raw_lms)
                    seq_landmarks.append(norm_vec)

            cap.release()
            if len(seq_landmarks) >= 12:
                extracted_sequences.append(np.array(seq_landmarks, dtype=np.float32))

        if extracted_sequences:
            np.save(output_file, np.array(extracted_sequences, dtype=object))
            print(f"  [SUKSES] {len(extracted_sequences)} sekuens disimpan ke {output_file.name}")
        else:
            print(f"  [WARN] Tidak ada sekuens valid yang diekstrak untuk '{class_name}'.")

    mp_extractor.close()
    print("\n[SELESAI] Pra-ekstraksi landmark selesai.")

if __name__ == "__main__":
    extract_dataset_landmarks()
