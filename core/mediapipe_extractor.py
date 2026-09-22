import cv2
import numpy as np

class MediaPipeExtractor:
    """
    Wrapper MediaPipe Hands untuk ekstraksi 21 koordinat landmark 3D.
    Mendukung dua mode:
    - Clean Mode (SBC/Production): Hanya mengekstrak koordinat numerik, tanpa menggambar di frame.
    - Debug Mode (PC/Testing): Menggambar 21 titik sendi dan garis skeleton tangan berwarna.
    """

    def __init__(self, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        import mediapipe as mp
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

    def extract(self, image, draw_debug=False):
        """
        Mengekstrak landmark dari frame citra.

        Args:
            image (numpy.ndarray): Frame citra BGR dari kamera
            draw_debug (bool): Jika True, gambar skeleton landmark pada image (in-place)

        Returns:
            tuple: (landmarks_found: bool, raw_landmarks: object or None, image: numpy.ndarray)
        """
        if image is None:
            return False, None, image

        # MediaPipe memerlukan format RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return False, None, image

        # Ambil tangan pertama yang terdeteksi
        hand_landmarks = results.multi_hand_landmarks[0]

        if draw_debug:
            # Menggambar 21 sendi dan garis koneksi pada frame
            self.mp_draw.draw_landmarks(
                image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )

        return True, hand_landmarks, image

    def close(self):
        """Menutup instance MediaPipe."""
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
