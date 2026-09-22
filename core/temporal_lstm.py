import numpy as np
import json
from pathlib import Path
from collections import deque

class TemporalLSTM:
    """
    LSTM Temporal Sequence Classifier.
    Menerima urutan vektor fitur spasial dari 1D-CNN sepanjang T frame (misal T=12)
    dan mengklasifikasikan ke dalam salah satu dari 14 kelas gestur.

    Mendukung:
    - TFLite Runtime (format paling efisien di Raspberry Pi 5).
    - ONNX Runtime.
    - Buffer FIFO internal untuk inferensi streaming frame-by-frame.
    """

    def __init__(self, model_path=None, labels_path=None, sequence_length=12):
        self.sequence_length = sequence_length
        self.model_path = model_path
        self.tflite_interpreter = None
        self.onnx_session = None

        # FIFO sequence buffer
        self.buffer = deque(maxlen=self.sequence_length)

        # Load label map
        if labels_path is None:
            labels_path = Path(__file__).resolve().parent.parent / "configs" / "labels.json"
        
        with open(labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.classes = data.get("classes", [])

        if model_path and Path(model_path).exists():
            path_str = str(model_path)
            if path_str.endswith(".tflite"):
                try:
                    import tflite_runtime.interpreter as tflite
                    self.tflite_interpreter = tflite.Interpreter(model_path=path_str)
                except ImportError:
                    import tensorflow as tf
                    self.tflite_interpreter = tf.lite.Interpreter(model_path=path_str)
                self.tflite_interpreter.allocate_tensors()
            elif path_str.endswith(".onnx"):
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(path_str, providers=['CPUExecutionProvider'])

    def add_feature(self, feature_vector):
        """Menambahkan satu vektor fitur 64-d ke buffer sekuens."""
        if feature_vector is not None:
            self.buffer.append(feature_vector)

    def is_ready(self):
        """Memeriksa apakah buffer sudah terisi penuh (T frame)."""
        return len(self.buffer) == self.sequence_length

    def predict(self):
        """
        Melakukan prediksi gestur berdasarkan sekuens di buffer saat ini.

        Returns:
            tuple: (predicted_class: str, confidence: float, probabilities: numpy.ndarray)
        """
        if not self.is_ready():
            return "menunggu_buffer", 0.0, np.zeros(len(self.classes), dtype=np.float32)

        sequence_data = np.array(self.buffer, dtype=np.float32)  # Shape: (12, 64)
        input_tensor = np.expand_dims(sequence_data, axis=0)     # Shape: (1, 12, 64)

        if self.tflite_interpreter is not None:
            input_details = self.tflite_interpreter.get_input_details()
            output_details = self.tflite_interpreter.get_output_details()
            self.tflite_interpreter.set_tensor(input_details[0]['index'], input_tensor)
            self.tflite_interpreter.invoke()
            probs = self.tflite_interpreter.get_tensor(output_details[0]['index'])[0]
        elif self.onnx_session is not None:
            input_name = self.onnx_session.get_inputs()[0].name
            outputs = self.onnx_session.run(None, {input_name: input_tensor})
            probs = outputs[0][0]
        else:
            # Baseline Dummy Predictor (sebelum model dilatih):
            # Menghasilkan prediksi deterministik berdasarkan rata-rata energi landmark
            probs = np.zeros(len(self.classes), dtype=np.float32)
            energy = np.mean(sequence_data)
            idx = int(abs(energy * 100)) % len(self.classes)
            probs[idx] = 0.85

        best_idx = int(np.argmax(probs))
        predicted_class = self.classes[best_idx] if best_idx < len(self.classes) else "unknown"
        confidence = float(probs[best_idx])

        return predicted_class, confidence, probs

    def reset(self):
        """Mengosongkan sequence buffer."""
        self.buffer.clear()
