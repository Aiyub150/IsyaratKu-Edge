import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Project Metadata
PROJECT_NAME = "IsyaratKu-edge"
VERSION = "1.0.0"

# Video Stream Configuration
CAMERA_INDEX = 0                  # Default webcam laptop / PC
ESP32_STREAM_URL = "http://192.168.1.100:81/stream"  # Default ESP32-CAM stream URL
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 20

# MediaPipe Configuration
MP_MAX_NUM_HANDS = 1
MP_MIN_DETECTION_CONFIDENCE = 0.6
MP_MIN_TRACKING_CONFIDENCE = 0.5
NUM_LANDMARKS = 21
LANDMARK_DIMENSIONS = 3          # x, y, z
LANDMARK_VECTOR_SIZE = NUM_LANDMARKS * LANDMARK_DIMENSIONS  # 63

# Sequence & Model Configuration
SEQUENCE_LENGTH = 12             # Jumlah frame per sequence temporal
NUM_CLASSES = 14
CONFIDENCE_THRESHOLD = 0.65       # Ambang batas kepercayaan prediksi
DEBOUNCE_FRAMES = 5              # Jumlah frame berturut-turut untuk stabilisasi gestur
COOLDOWN_SECONDS = 1.5           # Cooldown antar kata agar tidak berulang cepat

# File Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "dataset_raw"
LANDMARKS_CACHE_DIR = DATA_DIR / "landmarks_cache"
MODELS_DIR = DATA_DIR / "models"

YOLO_MODEL_PATH = MODELS_DIR / "yolov8n_hand.onnx"
SPATIAL_CNN_PATH = MODELS_DIR / "spatial_1dcnn.onnx"
TEMPORAL_LSTM_PATH = MODELS_DIR / "temporal_lstm.tflite"
LABELS_PATH = BASE_DIR / "configs" / "labels.json"
DATABASE_PATH = BASE_DIR / "data" / "isyaratku.db"
