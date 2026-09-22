import cv2
import time
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from configs.settings import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, ESP32_STREAM_URL

def test_camera_stream(source=CAMERA_INDEX):
    """
    Modul Pengujian 1: Uji Kestabilan & FPS Streaming Kamera.
    Mendukung webcam lokal (int) atau stream ESP32-CAM (URL).
    
    Menampilkan Metrik Performa:
    - Current FPS & Average FPS
    - Frame Resolution
    - Frame Acquisition Latency (ms)
    """
    print(f"[TEST STREAM] Membuka sumber video: {source}")
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka sumber video: {source}")
        return

    prev_time = time.time()
    frame_count = 0
    start_time = time.time()
    fps_history = []

    print("[INFO] Tekan 'q' di jendela preview untuk keluar.")

    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Gagal membaca frame.")
            break

        frame_count += 1
        curr_time = time.time()
        
        # Perhitungan Latensi & FPS
        frame_latency_ms = (curr_time - t_start) * 1000.0
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
        prev_time = curr_time
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        # RENDER DEBUG HUD (Heads-Up Display)
        h, w = frame.shape[:2]
        # Background box semi-transparan untuk metrik
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 110), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Teks Metrik Performa
        cv2.putText(frame, f"STREAM TEST - {PROJECT_NAME if 'PROJECT_NAME' in globals() else 'IsyaratKu'}", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f} (Avg: {avg_fps:.1f})", 
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Latency: {frame_latency_ms:.1f} ms", 
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Resolution: {w}x{h}", 
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("IsyaratKu-edge [TEST: Camera Stream FPS]", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    total_time = time.time() - start_time
    print(f"\n[HASIL TEST STREAM]")
    print(f"Total Frame : {frame_count}")
    print(f"Total Waktu : {total_time:.2f} s")
    print(f"Rata-rata FPS: {frame_count / total_time:.2f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Uji FPS Stream Kamera")
    parser.add_argument("--esp32", action="store_true", help="Gunakan stream URL ESP32-CAM dari settings.py")
    args = parser.parse_args()

    src = ESP32_STREAM_URL if args.esp32 else CAMERA_INDEX
    test_camera_stream(source=src)
