import cv2
import time
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

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

def run_pc_main(source=CAMERA_INDEX, debug_mode=False):
    """
    MAIN RUNNER UNTUK PC / LAPTOP:
    
    Mode:
    - Normal / Clean Mode (debug_mode=False): Tampilan visual bersih seperti di SBC, cocok untuk demonstrasi.
    - Debug Mode (debug_mode=True): Tampilan dengan HUD performa, skeleton, dan metrik latensi.
    """
    print("=" * 60)
    print(f"  ISYARATKU-EDGE: PC RUNNER (Debug Mode: {'ON' if debug_mode else 'OFF'})  ")
    print("=" * 60)

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
        print(f"[ERROR] Gagal membuka kamera: {source}")
        return

    print("[INFO] Tekan 'd' untuk toggle debug HUD, 'r' untuk reset kalimat, 'q' untuk keluar.")

    prev_time = time.time()
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # 1. YOLO Deteksi
        has_hand, bbox, frame = yolo.detect(frame, draw_debug=debug_mode)

        predicted_class = "none"
        confidence = 0.0

        if has_hand:
            # 2. MediaPipe Landmark
            found, raw_landmarks, frame = mp_extractor.extract(frame, draw_debug=debug_mode)
            if found:
                # 3. Normalisasi
                norm_vector = normalize_landmarks(raw_landmarks)

                # Deteksi Gestur 'Salah' SIBI untuk reset kalimat (Feedback #4 Poin 6)
                if sentence_builder.check_reset_gesture(norm_vector):
                    lstm_classifier.reset()
                    print("[EVENT RESET] Gestur SIBI 'Salah' terdeteksi -> Kalimat direset.")

                # 4. 1D-CNN & LSTM
                feat = cnn_encoder.encode(norm_vector)
                lstm_classifier.add_feature(feat)
                predicted_class, confidence, _ = lstm_classifier.predict()

                # 5. Sentence Builder & TTS (FSM Feedback #2)
                status = sentence_builder.process_gesture(predicted_class, confidence, threshold=CONFIDENCE_THRESHOLD)
                if status.get("word_to_speak"):
                    print(f"[TTS OUTPUT]: \"{status['word_to_speak']}\" (FSM: {status['state']})")
                    tts.speak(status["word_to_speak"])

        # Perhitungan FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        # Render Visual
        curr_sentence = sentence_builder._get_status(None, None, False)["sentence"]
        if debug_mode:
            # Tampilan HUD Lengkap
            cv2.putText(frame, f"FPS: {fps:.1f} | Conf: {confidence * 100:.0f}%", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Gestur: {predicted_class}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Kalimat: {curr_sentence if curr_sentence else '[Menunggu...]'}", (20, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            # Tampilan Bersih (Clean Subtitle)
            if curr_sentence:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h - 60), (w, h), (15, 15, 15), -1)
                cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
                cv2.putText(frame, curr_sentence, (30, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("IsyaratKu-edge [PC Runner]", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"[INFO] Debug HUD: {'AKTIF' if debug_mode else 'NONAKTIF'}")
        elif key == ord('r'):
            sentence_builder.reset_sentence()
            lstm_classifier.reset()
            print("[INFO] Kalimat di-reset.")

    cap.release()
    mp_extractor.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IsyaratKu-edge PC Runner")
    parser.add_argument("--debug", action="store_true", help="Mulai langsung dengan mode Debug HUD aktif")
    args = parser.parse_args()
    run_pc_main(debug_mode=args.debug)
