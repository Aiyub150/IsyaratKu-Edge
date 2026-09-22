import numpy as np
from pathlib import Path

class Spatial1DCNN:
    """
    1D-CNN Spatial Landmark Encoder.
    Menerima input vektor 21 koordinat landmark ternormalisasi (63 float)
    dan mengekstraksi vektor representasi spasial berdimensi 64 (spatial feature vector).

    Mendukung:
    - ONNX Runtime untuk inferensi ultra-cepat di SBC dan PC.
    - PyTorch / Keras untuk keperluan training di PC.
    """

    def __init__(self, model_path=None):
        self.model_path = model_path
        self.onnx_session = None

        if model_path and Path(model_path).exists():
            path_str = str(model_path)
            if path_str.endswith(".onnx"):
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(path_str, providers=['CPUExecutionProvider'])

    def encode(self, landmark_vector):
        """
        Mengekstrak fitur spasial dari vektor landmark 63-d.

        Args:
            landmark_vector (numpy.ndarray): Vektor 1D bentuk (63,)

        Returns:
            numpy.ndarray: Vektor fitur spasial 1D bentuk (64,)
        """
        if landmark_vector is None or len(landmark_vector) != 63:
            return np.zeros(64, dtype=np.float32)

        if self.onnx_session is not None:
            input_name = self.onnx_session.get_inputs()[0].name
            # Format input: (1, 63, 1) atau (1, 21, 3) sesuai rancangan training
            input_data = landmark_vector.reshape(1, 63, 1).astype(np.float32)
            outputs = self.onnx_session.run(None, {input_name: input_data})
            return outputs[0].flatten().astype(np.float32)

        # Baseline Dummy Encoder (jika model belum dilatih)
        # Mengembalikan representasi berbobot agar pipeline end-to-end dapat diuji
        dummy_feature = np.zeros(64, dtype=np.float32)
        dummy_feature[:63] = landmark_vector
        return dummy_feature
