import cv2
import numpy as np
from pathlib import Path

class YOLODetector:
    """
    Wrapper YOLOv8n untuk mendeteksi keberadaan objek tangan (hand presence).
    Mendukung format ONNX Runtime (ringan untuk SBC) dan format PyTorch/Ultralytics (.pt).

    Fitur:
    - Clean Mode (SBC/Production): Mengembalikan koordinat bounding box tanpa rendering visual.
    - Debug Mode (PC/Testing): Menggambar bounding box, label, dan confidence score pada frame.
    """

    def __init__(self, model_path=None, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.onnx_session = None
        self.pt_model = None

        if model_path and Path(model_path).exists():
            path_str = str(model_path)
            if path_str.endswith(".onnx"):
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(path_str, providers=['CPUExecutionProvider'])
            elif path_str.endswith(".pt"):
                from ultralytics import YOLO
                self.pt_model = YOLO(path_str)

    def detect(self, image, draw_debug=False):
        """
        Mendeteksi objek tangan pada frame citra.

        Args:
            image (numpy.ndarray): Frame BGR dari kamera
            draw_debug (bool): Jika True, gambar bounding box & info pada image

        Returns:
            tuple: (detected: bool, bbox: tuple (x1, y1, x2, y2) or None, image: numpy.ndarray)
        """
        if image is None:
            return False, None, image

        h, w = image.shape[:2]

        # Jika model belum dimuat (tahap baseline sebelum training), gunakan fallback full-frame
        if self.onnx_session is None and self.pt_model is None:
            # Fallback: Anggap seluruh frame adalah area ROI tangan
            return True, (0, 0, w, h), image

        detected = False
        best_bbox = None
        best_conf = 0.0

        if self.pt_model is not None:
            results = self.pt_model.predict(image, conf=self.confidence_threshold, verbose=False)
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                best_idx = int(boxes.conf.argmax())
                best_conf = float(boxes.conf[best_idx])
                xyxy = boxes.xyxy[best_idx].cpu().numpy()
                best_bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                detected = True

        elif self.onnx_session is not None:
            # ONNX Inference implementation
            input_name = self.onnx_session.get_inputs()[0].name
            img_resized = cv2.resize(image, (640, 640))
            img_transposed = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
            input_tensor = np.expand_dims(img_transposed, axis=0)
            
            outputs = self.onnx_session.run(None, {input_name: input_tensor})
            # Parsing output YOLOv8 ONNX
            # (Format [1, 5, 8400] -> x, y, w, h, conf)
            preds = outputs[0][0]
            scores = preds[4, :]
            valid_idx = np.where(scores > self.confidence_threshold)[0]
            if len(valid_idx) > 0:
                best_i = valid_idx[np.argmax(scores[valid_idx])]
                best_conf = float(scores[best_i])
                cx, cy, bw, bh = preds[0:4, best_i]
                x1 = int((cx - bw / 2) * (w / 640.0))
                y1 = int((cy - bh / 2) * (h / 640.0))
                x2 = int((cx + bw / 2) * (w / 640.0))
                y2 = int((cy + bh / 2) * (h / 640.0))
                best_bbox = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
                detected = True

        # Render debug visual jika diminta
        if draw_debug and detected and best_bbox:
            x1, y1, x2, y2 = best_bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Hand: {best_conf * 100:.1f}%"
            cv2.putText(image, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return detected, best_bbox, image
