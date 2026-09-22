import cv2
import time
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from configs.settings import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT
from core.mediapipe_extractor import MediaPipeExtractor

def test_mediapipe_fps(source=CAMERA_INDEX):
    """
    Modul Pengujian 3: Uji Ekstraksi & Tracking 21 Landmark MediaPipe.

    Menampilkan Metrik Performa:
    - MediaPipe Extraction Latency (ms)
    - Tracking FPS
    - Visualisasi 21 Landmark & Koneksi Sendi
    - Jumlah Landmark yang Berhasil Diekstrak (21/21)
    """
    print("[TEST MEDIAPIPE] Inisialisasi MediaPipe Hands...")
    mp_extractor = MediaPipeExtractor(min_detection_confidence=0.6, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka sumber video: {source}")
        return

    latency_history = []
    prev_time = time.time()
    print("[INFO] Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Uji Latensi Ekstraksi
        t0 = time.time()
        found, landmarks, frame = mp_extractor.extract(frame, draw_debug=True)
        t_ms = (time.time() - t0) * 1000.0

        latency_history.append(t_ms)
        if len(latency_history) > 30:
            latency_history.pop(0)
        avg_latency = sum(latency_history) / len(latency_history)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time

        # RENDER DEBUG HUD
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (330, 95), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, "MEDIAPIPE HANDS TEST [PC]", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, f"Latency: {t_ms:.1f} ms (Avg: {avg_latency:.1f} ms)", 
                    (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f} | Landmarks: {'21/21' if found else '0/21'}", 
                    (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if found else (0, 0, 255), 2)

        cv2.imshow("IsyaratKu-edge [TEST: MediaPipe FPS & Landmarks]", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    mp_extractor.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_mediapipe_fps()
