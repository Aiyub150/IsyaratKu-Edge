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

    def __init__(self, max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5, model_path=None):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.prev_primary_center = None

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

    def _select_primary_hand(self, all_hands):
        """
        Menentukan tangan primer (dominan) secara konsisten antar-frame.
        Mencegah jittering/berganti tangan acak saat kedua tangan terdeteksi (misal gestur 'belajar').
        """
        if not all_hands:
            return None, None
        if len(all_hands) == 1:
            landmarks = all_hands[0] if hasattr(all_hands[0], '__iter__') else all_hands[0].landmark
            cx = np.mean([lm.x for lm in landmarks])
            cy = np.mean([lm.y for lm in landmarks])
            self.prev_primary_center = (cx, cy)
            return landmarks, 0

        # Jika ada > 1 tangan, cari tangan yang paling dekat dengan posisi frame sebelumnya
        best_idx = 0
        best_dist = float('inf')

        for idx, hand in enumerate(all_hands):
            landmarks = hand if hasattr(hand, '__iter__') else hand.landmark
            cx = np.mean([lm.x for lm in landmarks])
            cy = np.mean([lm.y for lm in landmarks])

            if self.prev_primary_center is not None:
                dist = (cx - self.prev_primary_center[0]) ** 2 + (cy - self.prev_primary_center[1]) ** 2
            else:
                # Jika belum ada referensi sebelumnya: pilih tangan yang posisinya paling mendekati tengah frame
                dist = (cx - 0.5) ** 2 + (cy - 0.5) ** 2

            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        primary_hand = all_hands[best_idx]
        primary_landmarks = primary_hand if hasattr(primary_hand, '__iter__') else primary_hand.landmark
        cx = np.mean([lm.x for lm in primary_landmarks])
        cy = np.mean([lm.y for lm in primary_landmarks])
        self.prev_primary_center = (cx, cy)
        return primary_landmarks, best_idx

    def extract(self, image, draw_debug=False):
        """
        Mengekstrak landmark dari frame citra.
        Menangani multiple hands dan menstabilkan tangan primer.

        Args:
            image (numpy.ndarray): Frame citra BGR dari kamera
            draw_debug (bool): Jika True, gambar skeleton landmark pada image (in-place)

        Returns:
            tuple: (landmarks_found: bool, raw_landmarks: list/object or None, image: numpy.ndarray)
        """
        if image is None:
            return False, None, image

        h, w = image.shape[:2]
        all_hands = []

        if not self.use_tasks_api:
            # Legacy Solutions API
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            if results.multi_hand_landmarks:
                all_hands = results.multi_hand_landmarks
        else:
            # Modern Tasks API
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=img_rgb)
            results = self.detector.detect(mp_image)
            if results.hand_landmarks:
                all_hands = results.hand_landmarks

        if not all_hands:
            self.prev_primary_center = None
            return False, None, image

        primary_landmarks, primary_idx = self._select_primary_hand(all_hands)

        if draw_debug:
            for idx, hand in enumerate(all_hands):
                lms = hand if hasattr(hand, '__iter__') else hand.landmark
                is_primary = (idx == primary_idx)
                self._draw_landmarks_cv2(image, lms, w, h, is_primary=is_primary)

        return True, primary_landmarks, image

    def _draw_landmarks_cv2(self, image, landmarks, w, h, is_primary=True):
        """Gambar skeleton dan titik landmark langsung dengan OpenCV untuk performa maksimal."""
        coords = []
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            coords.append((cx, cy))

        # Garis koneksi sendi: Hijau neon untuk primer, abu-abu/cyan untuk sekunder
        line_color = (0, 230, 115) if is_primary else (180, 180, 180)
        dot_color = (0, 215, 255) if is_primary else (200, 200, 200)

        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(coords) and end_idx < len(coords):
                cv2.line(image, coords[start_idx], coords[end_idx], line_color, 2 if is_primary else 1, cv2.LINE_AA)

        for cx, cy in coords:
            cv2.circle(image, (cx, cy), 5 if is_primary else 3, dot_color, -1, cv2.LINE_AA)
            cv2.circle(image, (cx, cy), 5 if is_primary else 3, (0, 0, 0), 1, cv2.LINE_AA)

    def close(self):
        """Menutup instance MediaPipe."""
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
        if hasattr(self, 'detector') and self.detector:
            self.detector.close()
