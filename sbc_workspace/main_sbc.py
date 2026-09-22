import cv2
import time
import sys
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.settings import (
    ESP32_STREAM_URL, CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IsyaratKu-SBC")

def run_sbc_production(source=ESP32_STREAM_URL, headless=False):
    """
    MAIN PRODUCTION RUNNER (SBC / Raspberry Pi 5):
    
    Karakteristik Output Produksi:
    - CLEAN OUTPUT: Tanpa teks FPS, tanpa angka latensi, tanpa skeleton debug.
    - Zero Rendering Overhead: Menghemat 100% siklus CPU/GPU dari operasi rendering grafis.
    - Menampilkan frame video bersih dengan bar subtitle terjemahan yang elegan (atau mode headless).
    - Memicu keluaran suara Text-to-Speech secara otomatis saat kalimat selesai.
    - Terintegrasi dengan systemd untuk auto-restart dan auto-start on boot.
    """
    logger.info("==================================================")
    logger.info("  ISYARATKU-EDGE: SBC PRODUCTION RUNTIME (RPi 5)  ")
    logger.info("==================================================")

    # Inisialisasi komponen dalam mode Clean (draw_debug = False)
    logger.info("Memuat model inferensi ONNX / TFLite...")
    yolo = YOLODetector(model_path=YOLO_MODEL_PATH, confidence_threshold=0.5)
    mp_extractor = MediaPipeExtractor(min_detection_confidence=0.6, min_tracking_confidence=0.5)
    cnn_encoder = Spatial1DCNN(model_path=SPATIAL_CNN_PATH)
    lstm_classifier = TemporalLSTM(model_path=TEMPORAL_LSTM_PATH, labels_path=LABELS_PATH)
    sentence_builder = SentenceBuilder(labels_path=LABELS_PATH, debounce_frames=DEBOUNCE_FRAMES, cooldown_seconds=COOLDOWN_SECONDS)
    tts = TTSEngine()

    logger.info(f"Membuka stream input dari: {source}")
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        logger.warning(f"Stream {source} tidak tersedia. Mencoba fallback ke camera index 0...")
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            logger.error("Gagal membuka semua sumber video. Menghentikan layanan.")
            return

    logger.info("Layanan berjalan stabil. Mode: CLEAN PRODUCTION.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Frame drop / koneksi stream terputus. Mencoba membaca ulang...")
            time.sleep(0.5)
            continue

        h, w = frame.shape[:2]

        # 1. YOLO Hand Presence (Clean Mode: draw_debug = False)
        has_hand, bbox, _ = yolo.detect(frame, draw_debug=False)

        if has_hand:
            # 2. MediaPipe Landmarks (Clean Mode: draw_debug = False)
            found, raw_landmarks, _ = mp_extractor.extract(frame, draw_debug=False)

            if found:
                # 3. Normalisasi Landmark (63-d)
                norm_vector = normalize_landmarks(raw_landmarks)

                # 4. 1D-CNN Spatial Encoding (64-d)
                spatial_feat = cnn_encoder.encode(norm_vector)
                lstm_classifier.add_feature(spatial_feat)

                # 5. LSTM Temporal Classification
                predicted_class, confidence, _ = lstm_classifier.predict()

                # 6. Sentence Builder (FSM Feedback #2)
                status = sentence_builder.process_gesture(predicted_class, confidence, threshold=CONFIDENCE_THRESHOLD)

                # 7. Output Audio TTS per kata
                if status.get("word_to_speak"):
                    logger.info(f"[TTS OUTPUT]: \"{status['word_to_speak']}\" (FSM: {status['state']})")
                    tts.speak(status["word_to_speak"])

        # Tampilan Visual Bersih (Jika bukan headless mode)
        if not headless:
            curr_sentence = sentence_builder._get_status(None, False)["sentence"]
            if curr_sentence:
                # Subtitle Bar Bersih di Bawah (Hanya menampilkan kalimat, tanpa metrik performa)
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h - 60), (w, h), (15, 15, 15), -1)
                cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                cv2.putText(frame, curr_sentence, (30, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow("IsyaratKu-edge [SBC Production View]", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    mp_extractor.close()
    cv2.destroyAllWindows()
    logger.info("Layanan SBC dihentikan.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IsyaratKu-edge SBC Production Runner")
    parser.add_argument("--camera", type=int, default=None, help="Gunakan camera index lokal (misal 0)")
    parser.add_argument("--stream", type=str, default=None, help="Gunakan custom URL stream")
    parser.add_argument("--headless", action="store_true", help="Jalankan tanpa GUI window (cocok untuk systemd)")
    args = parser.parse_args()

    src = args.camera if args.camera is not None else (args.stream if args.stream else ESP32_STREAM_URL)
    run_sbc_production(source=src, headless=args.headless)
