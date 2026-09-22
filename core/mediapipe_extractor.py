import os
import urllib.request
import cv2
import numpy as np

# 21 Landmark Hand Connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "models", "hand_landmarker.task")


class MediaPipeExtractor:
    """
    Wrapper MediaPipe Hands untuk ekstraksi 21 koordinat landmark 3D.
    Mendukung MediaPipe 1.0+ (Tasks API) dan MediaPipe 0.10.x (Solutions API).
    
    Mode:
    - Clean Mode (SBC/Production): Ekstraksi koordinat numerik murni tanpa rendering.
    - Debug Mode (PC/Testing): Visualisasi 21 titik sendi dan garis skeleton tangan berwarna.
    """

    def __init__(self, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5, model_path=None):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_path = model_path or DEFAULT_MODEL_PATH

        import mediapipe as mp
        self.mp = mp

        # Deteksi apakah menggunakan Solutions API atau Tasks API
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.use_tasks_api = False
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
        else:
            self.use_tasks_api = True
            self._init_tasks_api()

    def _init_tasks_api(self):
        """Inisialisasi HandLandmarker dari MediaPipe Tasks API."""
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        # Pastikan file model ada, unduh otomatis jika belum ada
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            print(f"[MediaPipe] Mengunduh hand_landmarker.task ke {self.model_path}...")
            urllib.request.urlretrieve(MODEL_URL, self.model_path)
            print("[MediaPipe] Unduhan model selesai.")

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def extract(self, image, draw_debug=False):
        """
        Mengekstrak landmark dari frame citra.

        Args:
            image (numpy.ndarray): Frame citra BGR dari kamera
            draw_debug (bool): Jika True, gambar skeleton landmark pada image (in-place)

        Returns:
            tuple: (landmarks_found: bool, raw_landmarks: list/object or None, image: numpy.ndarray)
        """
        if image is None:
            return False, None, image

        h, w = image.shape[:2]

        if not self.use_tasks_api:
            # Legacy Solutions API
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            if not results.multi_hand_landmarks:
                return False, None, image
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks_list = hand_landmarks.landmark
        else:
            # Modern Tasks API
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=img_rgb)
            results = self.detector.detect(mp_image)
            if not results.hand_landmarks or len(results.hand_landmarks) == 0:
                return False, None, image
            landmarks_list = results.hand_landmarks[0]

        if draw_debug:
            self._draw_landmarks_cv2(image, landmarks_list, w, h)

        return True, landmarks_list, image

    def _draw_landmarks_cv2(self, image, landmarks, w, h):
        """Gambar skeleton dan titik landmark langsung dengan OpenCV untuk performa maksimal."""
        coords = []
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            coords.append((cx, cy))

        # Gambar garis koneksi antar sendi (Cyan / Neon Green)
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(coords) and end_idx < len(coords):
                pt1 = coords[start_idx]
                pt2 = coords[end_idx]
                cv2.line(image, pt1, pt2, (0, 230, 115), 2, cv2.LINE_AA)

        # Gambar titik sendi (Kuning emas dengan batas hitam)
        for cx, cy in coords:
            cv2.circle(image, (cx, cy), 5, (0, 215, 255), -1, cv2.LINE_AA)
            cv2.circle(image, (cx, cy), 5, (0, 0, 0), 1, cv2.LINE_AA)

    def close(self):
        """Menutup instance MediaPipe."""
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
        if hasattr(self, 'detector') and self.detector:
            self.detector.close()
