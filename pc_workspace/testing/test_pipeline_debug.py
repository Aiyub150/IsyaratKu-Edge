import cv2
import time
import sys
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from configs.settings import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    YOLO_MODEL_PATH, SPATIAL_CNN_PATH, TEMPORAL_LSTM_PATH,
    LABELS_PATH, CONFIDENCE_THRESHOLD, DEBOUNCE_FRAMES, COOLDOWN_SECONDS
)
from core.yolo_detector import YOLODetector
from core.mediapipe_extractor import MediaPipeExtractor
from core.landmark_normalizer import normalize_landmarks
from core.spatial_cnn import Spatial1DCNN
from core.temporal_lstm import TemporalLSTM
from core.sentence_builder import SentenceBuilder
from core.tts_engine import TTSEngine

def run_pipeline_debug_test(source=CAMERA_INDEX):
    """
    MODUL TESTING UTAMA (PC):
    Menjalankan FULL PIPELINE Machine Learning secara interaktif dengan
    Heads-Up Display (HUD) lengkap untuk analisis performa:

    Visual yang ditampilkan:
    1. Bounding Box deteksi tangan (YOLO).
    2. 21 titik skeleton tangan berwarna (MediaPipe Hands).
    3. HUD Metrik: FPS, latensi per tahap (YOLO ms, MP ms, Model ms, Total ms).
    4. Indikator status buffer State Machine (Subjek + Predikat).
    5. Confidence bar gestur yang terdeteksi.
    6. Keluaran audio TTS saat kalimat selesai.
    """
    print("=" * 60)
    print("   ISYARATKU-EDGE: FULL PIPELINE DEBUG & PERFORMANCE TEST   ")
    print("=" * 60)

    # Inisialisasi komponen
    print("[INIT] Memuat komponen pipeline...")
    yolo = YOLODetector(model_path=YOLO_MODEL_PATH, confidence_threshold=0.5)
    mp_extractor = MediaPipeExtractor(min_detection_confidence=0.6, min_tracking_confidence=0.5)
    cnn_encoder = Spatial1DCNN(model_path=SPATIAL_CNN_PATH)
    lstm_classifier = TemporalLSTM(model_path=TEMPORAL_LSTM_PATH, labels_path=LABELS_PATH)
    sentence_builder = SentenceBuilder(labels_path=LABELS_PATH, debounce_frames=DEBOUNCE_FRAMES, cooldown_seconds=COOLDOWN_SECONDS)
    tts = TTSEngine()

    # Muat daftar kelas untuk fitur developer capture (Feedback #2 Poin 4)
    import json
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels_meta = json.load(f)
        classes = labels_meta["classes"]
    
    target_class_idx = 0  # Default target: 'saya'
    raw_dataset_base = Path(__file__).resolve().parent.parent.parent / "data" / "dataset_raw"
    from collections import deque
    landmark_history = deque(maxlen=12)

    # Banner notifikasi capture & reset
    capture_banner_text = ""
    capture_banner_time = 0.0
    reset_banner_text = ""
    reset_banner_time = 0.0

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka kamera sumber: {source}")
        return

    prev_time = time.time()
    fps_history = []
    
    # Gambar panduan visual gestur SIBI resmi (docs/panduan_subjek.jpg & docs/panduan_predikat.jpg)
    docs_dir = Path(__file__).resolve().parent.parent.parent / "docs"
    guide_subjek_img = cv2.imread(str(docs_dir / "panduan_subjek.jpg"))
    guide_predikat_img = cv2.imread(str(docs_dir / "panduan_predikat.jpg"))
    help_page = 0  # 0: Kamera, 1: Panduan Subjek, 2: Panduan Predikat

    print("[INFO] Pipeline aktif.")
    print("  - Tekan 'C' untuk CAPTURE gestur aktif ke dataset.")
    print("  - Tekan 'TAB' / 'N' / 'P' atau angka 0-9 untuk memilih target kelas.")
    print("  - Tekan 'H' untuk melihat GAMBAR PANDUAN GESTUR SIBI (Subjek -> Predikat -> Tutup).")
    print("  - Tekan 'M' untuk toggle Direct Word Mode / FSM Mode.")
    print("  - Tekan 'R' untuk reset, 'ESC' / 'Q' untuk keluar.\n")

    while True:
        t_total_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontal agar seperti cermin (mirror preview)
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # 1. YOLO Hand Presence Check
        t_yolo_start = time.time()
        has_hand, bbox, frame = yolo.detect(frame, draw_debug=True)
        t_yolo_ms = (time.time() - t_yolo_start) * 1000.0

        t_mp_ms = 0.0
        t_model_ms = 0.0
        predicted_class = "tidak_ada_tangan"
        confidence = 0.0
        raw_landmarks = None

        if has_hand:
            # 2. MediaPipe Landmark Extraction (Debug Draw = True)
            t_mp_start = time.time()
            found_landmarks, raw_landmarks, frame = mp_extractor.extract(frame, draw_debug=True)
            t_mp_ms = (time.time() - t_mp_start) * 1000.0

            if found_landmarks:
                t_model_start = time.time()
                # 3. Normalisasi Koordinat Landmark (63 float)
                norm_vector = normalize_landmarks(raw_landmarks)
                landmark_history.append(norm_vector)

                # Deteksi Gestur 'Salah' SIBI untuk reset kalimat (Feedback #4 Poin 6)
                if sentence_builder.check_reset_gesture(norm_vector):
                    lstm_classifier.reset()
                    landmark_history.clear()
                    reset_banner_text = "[GESTUR SALAH (SIBI)] Kalimat Berhasil Direset!"
                    reset_banner_time = time.time()
                    print("[EVENT RESET] Gestur SIBI 'Salah' terdeteksi -> Kalimat direset.")

                # 4. 1D-CNN Spatial Feature Encoding (64 float)
                spatial_feat = cnn_encoder.encode(norm_vector)
                lstm_classifier.add_feature(spatial_feat)

                # 5. LSTM Temporal Classification
                predicted_class, confidence, probs = lstm_classifier.predict()
                t_model_ms = (time.time() - t_model_start) * 1000.0

                # 6. Sentence Builder (FSM Feedback #2 Poin 5)
                status = sentence_builder.process_gesture(predicted_class, confidence, threshold=CONFIDENCE_THRESHOLD)

                # 7. Trigger TTS per kata langsung saat diterima
                if status.get("word_to_speak"):
                    word = status["word_to_speak"]
                    print(f"[EVENT TTS] Kata Diterima ({status['state']}): \"{word}\" -> Mengucapkan suara...")
                    tts.speak(word)

        t_total_ms = (time.time() - t_total_start) * 1000.0
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time
        fps_history.append(fps)
        if len(fps_history) > 20:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        # ==========================================
        # RENDER DEBUG HUD & PERFORMANCE METRICS
        # ==========================================
        # 1. Panel Metrik Kiri Atas
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (360, 160), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.putText(frame, "ISYARATKU-EDGE [DEBUG HUD]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f} (Avg: {avg_fps:.1f})", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Total Latency: {t_total_ms:.1f} ms", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"  YOLOv8  : {t_yolo_ms:.1f} ms", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"  MediaPipe: {t_mp_ms:.1f} ms", (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"  CNN+LSTM (ONNX): {t_model_ms:.1f} ms", (20, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        target_class = classes[target_class_idx]
        target_dir = raw_dataset_base / target_class
        existing_samples = len(list(target_dir.glob("*.npy"))) if target_dir.exists() else 0
        cv2.putText(frame, f"Target Capture: [{target_class.upper()}] ({existing_samples} sampel)", 
                    (20, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 2. Panel Hasil Gestur & Kalimat Bawah
        overlay_bottom = frame.copy()
        cv2.rectangle(overlay_bottom, (10, h - 100), (w - 10, h - 10), (20, 20, 20), -1)
        cv2.addWeighted(overlay_bottom, 0.8, frame, 0.2, 0, frame)

        # Bar Confidence
        conf_color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
        bar_w = int((w - 240) * confidence)
        cv2.rectangle(frame, (120, h - 90), (120 + bar_w, h - 78), conf_color, -1)
        cv2.rectangle(frame, (120, h - 90), (w - 120, h - 78), (100, 100, 100), 1)
        cv2.putText(frame, f"Conf: {confidence * 100:.0f}%", (20, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Status FSM / Direct Mode & Kalimat Terkini (Feedback #3 Poin 8)
        if sentence_builder.fsm_mode:
            fsm_state = sentence_builder.current_state
            state_label = f"FSM: MENUNGGU SUBJEK" if fsm_state == "WAIT_SUBJECT" else f"FSM: MENUNGGU PREDIKAT ({sentence_builder.current_subject})"
        else:
            state_label = "MODE: DETEKSI LANGSUNG (Direct Word Mode)"

        curr_sentence = sentence_builder._get_status(None, None, False)["sentence"]
        if not curr_sentence:
            curr_sentence = f"[ {state_label} ]"

        cv2.putText(frame, state_label, (20, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 200, 0), 2)
        cv2.putText(frame, f"Kata/Kalimat: {curr_sentence}", (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(frame, f"Prediksi: {predicted_class} | [C] Capture -> [{target_class}] | [TAB] Ganti Target | [M] Mode FSM | [H] Bantuan", 
                    (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

        # Banner Notifikasi Capture (Jika developer menekan 'C')
        if (time.time() - capture_banner_time) < 3.0 and capture_banner_text:
            banner_overlay = frame.copy()
            cv2.rectangle(banner_overlay, (20, h // 2 - 25), (w - 20, h // 2 + 25), (0, 120, 0), -1)
            cv2.addWeighted(banner_overlay, 0.85, frame, 0.15, 0, frame)
            cv2.putText(frame, capture_banner_text, (35, h // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Banner Notifikasi Gestur 'Salah' Reset Kalimat (Feedback #4 Poin 6)
        if (time.time() - reset_banner_time) < 2.5 and reset_banner_text:
            reset_overlay = frame.copy()
            cv2.rectangle(reset_overlay, (20, h // 2 + 35), (w - 20, h // 2 + 85), (0, 0, 180), -1)
            cv2.addWeighted(reset_overlay, 0.85, frame, 0.15, 0, frame)
            cv2.putText(frame, reset_banner_text, (35, h // 2 + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        # Tampilkan Window Preview (Kamera atau Gambar Panduan Visual SIBI)
        if help_page == 1 and guide_subjek_img is not None:
            display_frame = cv2.resize(guide_subjek_img, (w, h))
            overlay_h = display_frame.copy()
            cv2.rectangle(overlay_h, (0, h - 50), (w, h), (15, 15, 15), -1)
            cv2.addWeighted(overlay_h, 0.85, display_frame, 0.15, 0, display_frame)
            cv2.putText(display_frame, "[PANDUAN SUBJEK (1/2)] Tekan 'H' untuk Panduan Predikat | Tekan 'ESC' untuk Tutup",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
            cv2.imshow("IsyaratKu-edge [PC TESTING - PERFORMANCE HUD]", display_frame)
        elif help_page == 2 and guide_predikat_img is not None:
            display_frame = cv2.resize(guide_predikat_img, (w, h))
            overlay_h = display_frame.copy()
            cv2.rectangle(overlay_h, (0, h - 50), (w, h), (15, 15, 15), -1)
            cv2.addWeighted(overlay_h, 0.85, display_frame, 0.15, 0, display_frame)
            cv2.putText(display_frame, "[PANDUAN PREDIKAT (2/2)] Tekan 'H' atau 'ESC' untuk Kembali ke Kamera",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
            cv2.imshow("IsyaratKu-edge [PC TESTING - PERFORMANCE HUD]", display_frame)
        else:
            cv2.imshow("IsyaratKu-edge [PC TESTING - PERFORMANCE HUD]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC: Tutup panduan atau keluar
            if help_page > 0:
                help_page = 0
                print("[INFO] Menutup panduan, kembali ke kamera.")
            else:
                break
        elif key == ord('q'):
            break
        elif key == ord('r'):
            sentence_builder.reset_sentence()
            lstm_classifier.reset()
            landmark_history.clear()
            print("[INFO] FSM Kalimat dan buffer di-reset.")
        elif key == ord('h'):
            help_page = (help_page + 1) % 3
            if help_page == 1:
                print("[INFO] Menampilkan Gambar Panduan Subjek.")
            elif help_page == 2:
                print("[INFO] Menampilkan Gambar Panduan Predikat.")
            else:
                print("[INFO] Menutup panduan, kembali ke kamera.")
        elif key == ord('m'):
            new_mode = not sentence_builder.fsm_mode
            sentence_builder.set_fsm_mode(new_mode)
            mode_name = "FSM (Subjek -> Predikat)" if new_mode else "Direct Word Mode (Deteksi Bebas Langsung)"
            print(f"[MODE BERUBAH] Mode sekarang: {mode_name}")
        elif key == 9 or key == ord('n'):  # TAB atau 'N': Target Berikutnya
            target_class_idx = (target_class_idx + 1) % len(classes)
            print(f"[TARGET GESTUR] Diubah ke: [{classes[target_class_idx]}]")
        elif key == ord('p'):  # 'P': Target Sebelumnya
            target_class_idx = (target_class_idx - 1) % len(classes)
            print(f"[TARGET GESTUR] Diubah ke: [{classes[target_class_idx]}]")
        elif ord('0') <= key <= ord('9'):  # Angka 0-9 untuk pilih cepat kelas 0-9
            idx = key - ord('0')
            if idx < len(classes):
                target_class_idx = idx
                print(f"[TARGET GESTUR] Diubah ke: [{classes[target_class_idx]}]")
        elif key == ord('c') or key == ord('s'):  # 'C' / 'S': CAPTURE GESTUR KE DATASET (Feedback #2 Poin 4)
            if len(landmark_history) == 12:
                target_dir = raw_dataset_base / target_class
                target_dir.mkdir(parents=True, exist_ok=True)
                timestamp_ms = int(time.time() * 1000)
                save_file = target_dir / f"seq_{timestamp_ms}.npy"
                seq_data = np.array(list(landmark_history), dtype=np.float32)
                np.save(save_file, seq_data)
                
                total_in_class = len(list(target_dir.glob("*.npy")))
                capture_banner_text = f"[CAPTURE SUKSES] 12 frame -> {target_class} (Total: {total_in_class}) | Prediksi: {predicted_class}"
                capture_banner_time = time.time()
                print(f"\n[DATASET CAPTURE] Berhasil menyimpan sekuens 12-frame!")
                print(f"  - File       : {save_file}")
                print(f"  - Target     : {target_class}")
                print(f"  - Prediksi ML: {predicted_class} (Conf: {confidence*100:.1f}%)")
                print(f"  - Total Sampel Kelas '{target_class}': {total_in_class}\n")
            else:
                capture_banner_text = f"[CAPTURE GAGAL] Buffer belum penuh ({len(landmark_history)}/12 frame). Tahan posisi tangan!"
                capture_banner_time = time.time()
                print(f"[WARN] Buffer baru terisi {len(landmark_history)}/12 frame. Tahan posisi tangan di depan kamera.")

    cap.release()
    mp_extractor.close()
    cv2.destroyAllWindows()
    print("[INFO] Pengujian selesai.")

if __name__ == "__main__":
    run_pipeline_debug_test()

