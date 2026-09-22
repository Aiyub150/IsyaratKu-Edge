import cv2
import time
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from configs.settings import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, YOLO_MODEL_PATH
from core.yolo_detector import YOLODetector

def test_yolo_latency(source=CAMERA_INDEX):
    """
    Modul Pengujian 2: Uji Latensi & Akurasi Deteksi Tangan YOLO.

    Menampilkan Metrik Performa:
    - YOLO Inference Time (ms)
    - Detection Confidence Score (%)
    - Bounding Box Tangan
    - FPS Pemrosesan Detektor
    """
    print(f"[TEST YOLO] Memuat model YOLO dari: {YOLO_MODEL_PATH}")
    yolo = YOLODetector(model_path=YOLO_MODEL_PATH, confidence_threshold=0.5)

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka sumber video: {source}")
        return

    latency_history = []
    print("[INFO] Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Uji Latensi Inferensi
        t0 = time.time()
        detected, bbox, frame = yolo.detect(frame, draw_debug=True)
        t_ms = (time.time() - t0) * 1000.0

        latency_history.append(t_ms)
        if len(latency_history) > 30:
            latency_history.pop(0)
        avg_latency = sum(latency_history) / len(latency_history)

        # RENDER DEBUG HUD
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 95), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, "YOLO DETECTOR TEST [PC]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, f"Latency: {t_ms:.1f} ms (Avg: {avg_latency:.1f} ms)", 
                    (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Hand Detected: {'YES' if detected else 'NO'}", 
                    (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if detected else (0, 0, 255), 2)

        cv2.imshow("IsyaratKu-edge [TEST: YOLO Latency]", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_yolo_latency()
