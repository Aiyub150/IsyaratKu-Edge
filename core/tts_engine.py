import os
import sys
import ctypes
import threading
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TTSEngine")

AUDIO_DIR = Path(__file__).resolve().parent.parent / "data" / "audio"

# Pemetaan fonetik Bahasa Indonesia untuk engine suara berbahasa asing/Inggris (fallback)
INDONESIAN_PHONETIC_MAP = {
    "saya": "sah yah",
    "kamu": "kah moo",
    "anda": "ahn dah",
    "kami": "kah mee",
    "kita": "kee tah",
    "dia": "dee ah",
    "mereka": "meh reh kah",
    "makan": "mah kahn",
    "minum": "mee noom",
    "tidur": "tee door",
    "belajar": "buh lah jar",
    "bekerja": "buh kehr jah",
    "berjalan": "buh r jah lahn",
    "membaca": "mem bah chah"
}

class TTSEngine:
    """
    Wrapper Text-to-Speech lokal non-blocking dengan dukungan:
    1. Native Indonesian Audio Clips (data/audio/): Suara penutur asli Bahasa Indonesia (100% natural, 0 aksen asing).
    2. pyttsx3 Engine (Offline Fallback): Deteksi otomatis voice ID bahasa Indonesia / transliterasi fonetik cerdas.
    """

    def __init__(self, rate=140, volume=1.0):
        self.rate = rate
        self.volume = volume
        self.engine = None
        self.is_indonesian_voice = False
        self._lock = threading.Lock()
        
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", self.rate)
            self.engine.setProperty("volume", self.volume)
            
            # Cari profil suara Bahasa Indonesia
            voices = self.engine.getProperty("voices") or []
            for v in voices:
                v_name = (getattr(v, "name", "") or "").lower()
                v_id = (getattr(v, "id", "") or "").lower()
                if any(kw in v_name or kw in v_id for kw in ["indonesia", "id-id", "id_id", "andika", "gadis"]):
                    self.engine.setProperty("voice", v.id)
                    self.is_indonesian_voice = True
                    logger.info(f"Suara Bahasa Indonesia terdeteksi & dipilih: {v.name}")
                    break

            if not self.is_indonesian_voice:
                logger.info("Suara lokal Bahasa Indonesia tidak ditemukan di OS. Mengutamakan native audio clips & adaptor fonetik.")

            logger.info("TTSEngine berhasil diinisialisasi.")
        except Exception as e:
            logger.warning(f"pyttsx3 tidak dapat diinisialisasi: {e}. Menggunakan audio clips & fallback logger.")

    def _get_audio_clip_path(self, word):
        """Mencari file audio MP3 penutur asli Indonesia di data/audio/."""
        clean = word.lower().replace(".", "").replace(",", "").strip()
        candidate = AUDIO_DIR / f"{clean}.mp3"
        if candidate.exists():
            return candidate
        return None

    def _play_audio_clip(self, audio_path):
        """Memutar file MP3 secara native di Windows/Linux."""
        try:
            if sys.platform == "win32":
                mci = ctypes.windll.winmm.mciSendStringW
                alias = f"clip_{threading.get_ident()}"
                mci(f'open "{audio_path}" type mpegvideo alias {alias}', None, 0, None)
                mci(f'play {alias} wait', None, 0, None)
                mci(f'close {alias}', None, 0, None)
                return True
            else:
                # Linux / SBC (mpg123 / aplay / ffplay)
                os.system(f"mpg123 -q '{audio_path}' 2>/dev/null || aplay -q '{audio_path}' 2>/dev/null")
                return True
        except Exception as e:
            logger.error(f"Gagal memutar native audio clip: {e}")
            return False

    def _prepare_text_for_speech(self, text):
        """Menyesuaikan teks agar dilafalkan dengan aksen Indonesia yang jelas."""
        if self.is_indonesian_voice:
            return text
        
        clean_text = text.lower().replace(".", "").replace(",", "").strip()
        words = clean_text.split()
        converted = [INDONESIAN_PHONETIC_MAP.get(w, w) for w in words]
        return " ".join(converted)

    def speak(self, text):
        """
        Mengucapkan teks secara asinkronus (non-blocking thread)
        agar tidak memblokir frame processing inferensi.
        """
        if not text or not text.strip():
            return

        thread = threading.Thread(target=self._speak_sync, args=(text.strip(),), daemon=True)
        thread.start()

    def _speak_sync(self, text):
        with self._lock:
            clean_text = text.lower().replace(".", "").replace(",", "").strip()
            words = clean_text.split()

            # 1. Prioritas Utama: Periksa ketersediaan audio native per-kata di data/audio/*.mp3
            all_clips = [self._get_audio_clip_path(w) for w in words]
            if all(clip is not None for clip in all_clips):
                for w, clip in zip(words, all_clips):
                    logger.info(f"[AUDIO OUTPUT - NATIVE INDONESIA]: \"{w}\" ({clip.name})")
                    self._play_audio_clip(clip)
                return

            # 2. Fallback: Gunakan pyttsx3 dengan adaptor fonetik
            speech_text = self._prepare_text_for_speech(text)
            logger.info(f"[AUDIO OUTPUT - TTS FALLBACK]: \"{text}\" (fonetik: \"{speech_text}\")")
            if self.engine:
                try:
                    self.engine.say(speech_text)
                    self.engine.runAndWait()
                except Exception as e:
                    logger.error(f"Gagal memutar audio TTS: {e}")

