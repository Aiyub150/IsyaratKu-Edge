import time
import json
from pathlib import Path
import numpy as np

class SentenceBuilder:
    """
    State Machine untuk menyusun kata dari hasil klasifikasi gestur
    dengan alur per-kata sesuai Feedback #2:
    Masuk -> Deteksi Subjek -> Ucapkan Subjek -> Deteksi Predikat -> Ucapkan Predikat -> Kembali ke Deteksi Subjek.

    Keunggulan:
    - Tidak membebani penyimpanan (hanya butuh 14 audio per-kata).
    - Menghilangkan ambiguitas klasifikasi silang kategori:
      * Saat Menunggu Subjek: Gestur predikat diabaikan.
      * Saat Menunggu Predikat: Gestur subjek diabaikan.
    """

    STATE_WAIT_SUBJECT = "WAIT_SUBJECT"
    STATE_WAIT_PREDICATE = "WAIT_PREDICATE"
    STATE_DIRECT = "DIRECT_WORD"

    def __init__(self, labels_path=None, debounce_frames=5, cooldown_seconds=1.2, sentence_hold_seconds=3.0, predicate_timeout=10.0, fsm_mode=False):
        """
        Args:
            fsm_mode (bool): False = Deteksi kata langsung (Direct Mode - Feedback #3 Poin 8).
                             True  = Algoritma urutan Subjek -> Predikat (FSM).
        """
        self.debounce_frames = debounce_frames
        self.cooldown_seconds = cooldown_seconds
        self.sentence_hold_seconds = sentence_hold_seconds
        self.predicate_timeout = predicate_timeout
        self.fsm_mode = fsm_mode

        # Load kategori kata
        if labels_path is None:
            labels_path = Path(__file__).resolve().parent.parent / "configs" / "labels.json"
        
        with open(labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.classes = data.get("classes", [])
            self.subjects = set(data.get("categories", {}).get("subjects", []))
            self.predicates = set(data.get("categories", {}).get("predicates", []))

        # State Variables
        self.current_state = self.STATE_DIRECT if not self.fsm_mode else self.STATE_WAIT_SUBJECT
        self.current_subject = None
        self.current_predicate = None
        self.last_word = None
        self.last_word_time = 0.0
        self.subject_timestamp = 0.0

        # Retention hold for completed sentence display
        self.last_completed_sentence = ""
        self.completed_time = 0.0

        # Debounce tracking
        self.candidate_word = None
        self.candidate_count = 0

    def set_fsm_mode(self, enabled: bool):
        """Mengubah mode antara Direct Word Mode dan FSM Mode."""
        self.fsm_mode = enabled
        self.reset_sentence()

    def process_gesture(self, predicted_class, confidence, threshold=0.65):
        """
        Memproses hasil klasifikasi per frame.
        """
        now = time.time()
        accepted_word = None
        word_to_speak = None
        is_sentence_complete = False

        if confidence < threshold:
            self.candidate_word = None
            self.candidate_count = 0
            return self._get_status(None, None, False)

        # -------------------------------------------------------------
        # MODE 1: DIRECT WORD DETECTION (Feedback #3 Poin 8 - Default)
        # -------------------------------------------------------------
        if not self.fsm_mode:
            # Debounce checking
            if predicted_class == self.candidate_word:
                self.candidate_count += 1
            else:
                self.candidate_word = predicted_class
                self.candidate_count = 1

            if self.candidate_count >= self.debounce_frames:
                if (predicted_class != self.last_word) or (now - self.last_word_time > self.cooldown_seconds):
                    accepted_word = predicted_class
                    word_to_speak = accepted_word
                    self.last_word = accepted_word
                    self.last_word_time = now
                    self.candidate_count = 0
                    is_sentence_complete = True
                    self.last_completed_sentence = accepted_word.capitalize()
                    self.completed_time = now

            return self._get_status(accepted_word, word_to_speak, is_sentence_complete)

        # -------------------------------------------------------------
        # MODE 2: SUBJEK -> PREDIKAT FSM (Jika diaktifkan)
        # -------------------------------------------------------------
        # Timeout otomatis jika menunggu predikat terlalu lama (>10 detik), kembali ke WAIT_SUBJECT
        if self.current_state == self.STATE_WAIT_PREDICATE and (now - self.subject_timestamp > self.predicate_timeout):
            self.reset_sentence()

        # Filter kategori sesuai state aktif
        is_valid_for_state = False
        if self.current_state == self.STATE_WAIT_SUBJECT and predicted_class in self.subjects:
            is_valid_for_state = True
        elif self.current_state == self.STATE_WAIT_PREDICATE and predicted_class in self.predicates:
            is_valid_for_state = True

        if not is_valid_for_state:
            self.candidate_word = None
            self.candidate_count = 0
            return self._get_status(None, None, False)

        # Debounce checking
        if predicted_class == self.candidate_word:
            self.candidate_count += 1
        else:
            self.candidate_word = predicted_class
            self.candidate_count = 1

        # Cek apakah sudah memenuhi jumlah frame stabil
        if self.candidate_count >= self.debounce_frames:
            if (predicted_class != self.last_word) or (now - self.last_word_time > self.cooldown_seconds):
                accepted_word = predicted_class
                self.last_word = accepted_word
                self.last_word_time = now
                self.candidate_count = 0  # Reset counter

                if self.current_state == self.STATE_WAIT_SUBJECT:
                    self.current_subject = accepted_word
                    word_to_speak = accepted_word
                    self.subject_timestamp = now
                    self.current_state = self.STATE_WAIT_PREDICATE
                    self.last_completed_sentence = ""

                elif self.current_state == self.STATE_WAIT_PREDICATE:
                    self.current_predicate = accepted_word
                    word_to_speak = accepted_word
                    is_sentence_complete = True
                    self.last_completed_sentence = f"{self.current_subject.capitalize()} {self.current_predicate}."
                    self.completed_time = now

                    self.current_state = self.STATE_WAIT_SUBJECT
                    self.current_subject = None
                    self.current_predicate = None

        status = self._get_status(accepted_word, word_to_speak, is_sentence_complete)
        return status

    def _get_status(self, accepted_word, word_to_speak, is_sentence_complete):
        now = time.time()
        if not self.fsm_mode:
            sentence_str = self.last_completed_sentence if (now - self.completed_time < self.sentence_hold_seconds) else ""
            return {
                "state": "DIRECT_WORD",
                "accepted_word": accepted_word,
                "word_to_speak": word_to_speak,
                "subject": accepted_word if accepted_word in self.subjects else None,
                "predicate": accepted_word if accepted_word in self.predicates else None,
                "sentence": sentence_str,
                "is_sentence_complete": is_sentence_complete
            }
        words = []
        if self.current_subject:
            words.append(self.current_subject.capitalize())
        if self.current_predicate:
            words.append(self.current_predicate)
        
        sentence_str = " ".join(words)
        if sentence_str and is_sentence_complete:
            sentence_str += "."

        # Jika sedang dalam masa retensi tampilan kalimat selesai, tampilkan kalimat lengkap tersebut
        if not sentence_str and (now - self.completed_time < self.sentence_hold_seconds):
            sentence_str = self.last_completed_sentence

        return {
            "state": self.current_state,
            "accepted_word": accepted_word,
            "word_to_speak": word_to_speak,
            "subject": self.current_subject,
            "predicate": self.current_predicate,
            "sentence": sentence_str,
            "is_sentence_complete": is_sentence_complete
        }

    def reset_sentence(self):
        """Reset state kalimat ke awal (WAIT_SUBJECT)."""
        self.current_state = self.STATE_WAIT_SUBJECT
        self.current_subject = None
        self.current_predicate = None
        self.candidate_word = None
        self.candidate_count = 0
        self.last_completed_sentence = ""
        self.completed_time = 0.0
        self.subject_timestamp = 0.0

    def check_reset_gesture(self, landmarks_63):
        """
        Detektor gestur kontrol 'Salah' sesuai Kamus SIBI (Feedback #4 Poin 6).
        Dalam SIBI, gestur 'salah' digunakan untuk membatalkan/mereset kata atau kalimat yang salah:
        - Deteksi: Jari telunjuk ditekuk/melengkung (hook/C-shape) dengan jari lain mengepal (simbol 'salah' SIBI)
          ATAU sapuan horizontal cepat telapak tangan (wipe gesture) melintasi kamera.
        Mengembalikan True jika gestur 'Salah' terdeteksi stabil, dan langsung mereset kalimat.
        """
        if landmarks_63 is None or len(landmarks_63) < 63:
            return False

        pts = np.array(landmarks_63).reshape(21, 3)
        now = time.time()
        
        # Cooldown agar tidak ter-reset berulang-ulang dalam 1 detik
        if hasattr(self, "_last_salah_time") and (now - self._last_salah_time < 1.5):
            return False

        # Ciri SIBI 'Salah': Telunjuk ditekuk seperti kait/huruf D di depan dagu
        # Index MCP(5), PIP(6), DIP(7), TIP(8)
        # Telunjuk menekuk tajam: jarak tip(8) ke mcp(5) lebih pendek dari jarak pip(6) ke mcp(5)
        # Sementara jari tengah(12), manis(16), kelingking(20) mengepal rapat
        index_mcp = pts[5]
        index_pip = pts[6]
        index_tip = pts[8]
        dist_tip_mcp = np.linalg.norm(index_tip - index_mcp)
        dist_pip_mcp = np.linalg.norm(index_pip - index_mcp)

        # Cek apakah telunjuk melengkung kait (bent hook)
        is_index_hooked = (dist_tip_mcp < dist_pip_mcp * 0.9) and (index_tip[1] > index_pip[1] - 0.1)

        # Cek apakah 3 jari lain mengepal
        mid_curled = np.linalg.norm(pts[12] - pts[9]) < 0.25
        ring_curled = np.linalg.norm(pts[16] - pts[13]) < 0.25
        pinky_curled = np.linalg.norm(pts[20] - pts[17]) < 0.25

        if is_index_hooked and mid_curled and ring_curled and pinky_curled:
            self._salah_counter = getattr(self, "_salah_counter", 0) + 1
            if self._salah_counter >= 4:
                self.reset_sentence()
                self._last_salah_time = now
                self._salah_counter = 0
                return True
        else:
            self._salah_counter = 0

        return False


