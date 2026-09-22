import time
import json
from pathlib import Path

class SentenceBuilder:
    """
    State Machine sederhana untuk menyusun kata dari hasil klasifikasi gestur
    menjadi kalimat dengan pola: SUBJEK + PREDIKAT.

    Fitur:
    - Debounce filter: Memerlukan gestur terdeteksi stabil selama N frame.
    - Cooldown: Mencegah pemicuan berulang dari kata yang sama dalam jeda waktu singkat.
    - Automatic Sentence Complete: Memanggil callback saat Subjek + Predikat lengkap.
    """

    STATE_EMPTY = "EMPTY"
    STATE_SUBJECT_RECEIVED = "SUBJECT_RECEIVED"
    STATE_SENTENCE_READY = "SENTENCE_READY"

    def __init__(self, labels_path=None, debounce_frames=5, cooldown_seconds=1.5):
        self.debounce_frames = debounce_frames
        self.cooldown_seconds = cooldown_seconds

        # Load kategori kata
        if labels_path is None:
            labels_path = Path(__file__).resolve().parent.parent / "configs" / "labels.json"
        
        with open(labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.subjects = set(data.get("categories", {}).get("subjects", []))
            self.predicates = set(data.get("categories", {}).get("predicates", []))

        # State Variables
        self.current_state = self.STATE_EMPTY
        self.current_subject = None
        self.current_predicate = None
        self.last_word = None
        self.last_word_time = 0.0

        # Debounce tracking
        self.candidate_word = None
        self.candidate_count = 0

    def process_gesture(self, predicted_class, confidence, threshold=0.65):
        """
        Memproses hasil klasifikasi per frame.

        Args:
            predicted_class (str): Nama kelas gesture (misal 'saya', 'makan')
            confidence (float): Nilai kepercayaan model (0.0 - 1.0)
            threshold (float): Batas minimum kepercayaan

        Returns:
            dict: Status pemrosesan kalimat {
                'state': str,
                'accepted_word': str or None,
                'sentence': str,
                'is_sentence_complete': bool
            }
        """
        now = time.time()
        accepted_word = None
        is_sentence_complete = False

        if confidence < threshold:
            self.candidate_word = None
            self.candidate_count = 0
            return self._get_status(None, False)

        # 1. Debounce checking
        if predicted_class == self.candidate_word:
            self.candidate_count += 1
        else:
            self.candidate_word = predicted_class
            self.candidate_count = 1

        # Cek apakah sudah memenuhi jumlah frame stabil
        if self.candidate_count >= self.debounce_frames:
            # 2. Cooldown checking (kata yang sama tidak boleh berulang dalam interval singkat)
            if (predicted_class != self.last_word) or (now - self.last_word_time > self.cooldown_seconds):
                accepted_word = predicted_class
                self.last_word = accepted_word
                self.last_word_time = now
                self.candidate_count = 0  # Reset counter

                # 3. State Machine Transition
                if accepted_word in self.subjects:
                    # Subjek baru diterima (menggantikan subjek lama jika belum ada predikat)
                    self.current_subject = accepted_word
                    self.current_state = self.STATE_SUBJECT_RECEIVED
                elif accepted_word in self.predicates:
                    if self.current_state == self.STATE_SUBJECT_RECEIVED:
                        # Pasangan Subjek + Predikat lengkap!
                        self.current_predicate = accepted_word
                        self.current_state = self.STATE_SENTENCE_READY
                        is_sentence_complete = True
                    else:
                        # Predikat muncul sebelum subjek: Abaikan atau jadikan kata tunggal
                        pass

        status = self._get_status(accepted_word, is_sentence_complete)
        
        # Jika kalimat sudah lengkap, otomatis reset state agar siap untuk kalimat berikutnya
        if is_sentence_complete:
            self.reset_sentence()

        return status

    def _get_status(self, accepted_word, is_sentence_complete):
        words = []
        if self.current_subject:
            words.append(self.current_subject.capitalize())
        if self.current_predicate:
            words.append(self.current_predicate)
        
        sentence_str = " ".join(words)
        if sentence_str and is_sentence_complete:
            sentence_str += "."

        return {
            "state": self.current_state,
            "accepted_word": accepted_word,
            "subject": self.current_subject,
            "predicate": self.current_predicate,
            "sentence": sentence_str,
            "is_sentence_complete": is_sentence_complete
        }

    def reset_sentence(self):
        """Reset state kalimat ke awal (EMPTY)."""
        self.current_state = self.STATE_EMPTY
        self.current_subject = None
        self.current_predicate = None
        self.candidate_word = None
        self.candidate_count = 0
