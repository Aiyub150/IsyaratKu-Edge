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

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka kamera sumber: {source}")
        return

    prev_time = time.time()
    fps_history = []
    show_help = False
    print("[INFO] Pipeline aktif. Tekan 'h' untuk bantuan gestur, 'r' untuk reset, 'q' untuk keluar.\n")

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

                # 4. 1D-CNN Spatial Feature Encoding (64 float)
                spatial_feat = cnn_encoder.encode(norm_vector)
                lstm_classifier.add_feature(spatial_feat)

                # 5. LSTM Temporal Classification
                predicted_class, confidence, probs = lstm_classifier.predict()
                t_model_ms = (time.time() - t_model_start) * 1000.0

                # 6. Sentence Builder
                status = sentence_builder.process_gesture(predicted_class, confidence, threshold=CONFIDENCE_THRESHOLD)

                # 7. Trigger TTS jika kalimat lengkap
                if status["is_sentence_complete"]:
                    print(f"[EVENT] Kalimat Selesai: \"{status['sentence']}\"")
                    tts.speak(status["sentence"])

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
        cv2.rectangle(overlay, (10, 10), (330, 150), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.putText(frame, "ISYARATKU-EDGE [DEBUG HUD]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f} (Avg: {avg_fps:.1f})", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Total Latency: {t_total_ms:.1f} ms", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"  YOLOv8  : {t_yolo_ms:.1f} ms", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"  MediaPipe: {t_mp_ms:.1f} ms", (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"  1D-CNN+LSTM: {t_model_ms:.1f} ms", (20, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(frame, f"Buffer: {len(lstm_classifier.buffer)}/{lstm_classifier.sequence_length}", 
                    (20, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)

        # 2. Panel Hasil Gestur & Kalimat Bawah
        overlay_bottom = frame.copy()
        cv2.rectangle(overlay_bottom, (10, h - 90), (w - 10, h - 10), (20, 20, 20), -1)
        cv2.addWeighted(overlay_bottom, 0.8, frame, 0.2, 0, frame)

        # Bar Confidence
        conf_color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
        bar_w = int((w - 240) * confidence)
        cv2.rectangle(frame, (120, h - 80), (120 + bar_w, h - 68), conf_color, -1)
        cv2.rectangle(frame, (120, h - 80), (w - 120, h - 68), (100, 100, 100), 1)
        cv2.putText(frame, f"Conf: {confidence * 100:.0f}%", (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Teks Kalimat Terkini
        curr_sentence = sentence_builder._get_status(None, False)["sentence"]
        if not curr_sentence:
            curr_sentence = "[ Menunggu gestur Subjek... ]"
        cv2.putText(frame, f"Kalimat: {curr_sentence}", (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(frame, f"Gestur: {predicted_class} | Tekan 'H' untuk Bantuan Gestur", (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # 3. Modal Overlay Bantuan Gestur (Toggle 'H')
        if show_help:
            help_overlay = frame.copy()
            cv2.rectangle(help_overlay, (40, 30), (w - 40, h - 30), (15, 15, 20), -1)
            cv2.addWeighted(help_overlay, 0.9, frame, 0.1, 0, frame)

            cv2.putText(frame, "PANDUAN BENTUK GESTUR (Tekan 'H' untuk Tutup)", (60, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            # Kolom Kiri: Subjek
            cv2.putText(frame, "[SUBJEK]", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            subjek_help = [
                "1. saya   : Telunjuk tunjuk ke dada",
                "2. kamu   : Telunjuk lurus ke depan",
                "3. anda   : Telapak tangan terbuka sopan",
                "4. kami   : Tangan 'C' melengkung ke dada",
                "5. kita   : Gerakan melingkar di dada",
                "6. dia    : Telunjuk tunjuk ke samping",
                "7. mereka : Sapuan tangan ke samping"
            ]
            for i, line in enumerate(subjek_help):
                cv2.putText(frame, line, (60, 130 + (i * 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

            # Kolom Kanan: Predikat
            cv2.putText(frame, "[PREDIKAT]", (340, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            predikat_help = [
                "1. makan   : Jari menguncup di mulut",
                "2. minum   : Ibu jari tegak ke bibir (cangkir)",
                "3. tidur   : Telapak tangan bantal di pipi",
                "4. belajar : Buka kedua telapak (buku)",
                "5. bekerja : Tangan kepal mengetuk bawah",
                "6. berjalan: 2 jari (telunjuk+tengah) melangkah",
                "7. membaca : Telunjuk telusuri telapak datar"
            ]
            for i, line in enumerate(predikat_help):
                cv2.putText(frame, line, (340, 130 + (i * 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

            cv2.putText(frame, "Aturan: Tahan Subjek (0.5s) -> Tahan Predikat (0.5s) -> Kalimat Jadi + Suara TTS",
                        (60, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 255, 100), 1)

        # Tampilkan Window Preview
        cv2.imshow("IsyaratKu-edge [PC TESTING - PERFORMANCE HUD]", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            sentence_builder.reset_sentence()
            lstm_classifier.reset()
            print("[INFO] Kalimat dan buffer di-reset.")
        elif key == ord('h'):
            show_help = not show_help

    cap.release()
    mp_extractor.close()
    cv2.destroyAllWindows()
    print("[INFO] Pengujian selesai.")

if __name__ == "__main__":
    run_pipeline_debug_test()
