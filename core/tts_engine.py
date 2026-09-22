import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TTSEngine")

class TTSEngine:
    """
    Wrapper Text-to-Speech lokal non-blocking.
    Menggunakan pyttsx3 untuk sintesis offline.
    Jika pyttsx3 tidak tersedia / gagal, fallback ke console logger tanpa membuat sistem crash.
    """

    def __init__(self, rate=150, volume=1.0):
        self.rate = rate
        self.volume = volume
        self.engine = None
        self._lock = threading.Lock()
        
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", self.rate)
            self.engine.setProperty("volume", self.volume)
            logger.info("pyttsx3 TTS Engine berhasil diinisialisasi.")
        except Exception as e:
            logger.warning(f"pyttsx3 tidak dapat diinisialisasi: {e}. Menggunakan fallback logger.")

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
            logger.info(f"[AUDIO OUTPUT]: \"{text}\"")
            if self.engine:
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception as e:
                    logger.error(f"Gagal memutar audio TTS: {e}")
