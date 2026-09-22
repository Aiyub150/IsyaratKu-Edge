"""
01_prepare_dataset.py
Generator & Preprocessor Dataset Landmark untuk 14 Kelas SIBI/BISINDO IsyaratKu-Edge.
Menghasilkan representasi vektor landmark spasial (63,) dan urutan sekuensial temporal (12, 63)
lengkap dengan augmentasi rotasi, skala, jitter, dan variasi kecepatan gerak.
"""

import os
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "labels.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "landmarks_cache")

def load_labels():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["classes"]

def get_base_hand_pose(class_idx):
    """
    Menghasilkan pose dasar 21 landmark 3D (x, y, z) untuk masing-masing dari 14 kelas.
    Struktur titik:
    0: Wrist
    1-4: Thumb (CMC, MCP, IP, Tip)
    5-8: Index (MCP, PIP, DIP, Tip)
    9-12: Middle (MCP, PIP, DIP, Tip)
    13-16: Ring (MCP, PIP, DIP, Tip)
    17-20: Pinky (MCP, PIP, DIP, Tip)
    """
    # Base open palm facing camera
    pose = np.array([
        [0.0, 0.0, 0.0],       # 0: Wrist
        [-0.15, -0.1, -0.05],  # 1
        [-0.25, -0.2, -0.08],  # 2
        [-0.32, -0.3, -0.1],   # 3
        [-0.38, -0.4, -0.12],  # 4: Thumb Tip
        [-0.1, -0.4, 0.0],     # 5: Index MCP
        [-0.12, -0.6, 0.0],    # 6
        [-0.13, -0.75, 0.0],   # 7
        [-0.14, -0.9, 0.0],    # 8: Index Tip
        [0.0, -0.42, 0.0],     # 9: Middle MCP
        [0.0, -0.65, 0.0],     # 10
        [0.0, -0.82, 0.0],     # 11
        [0.0, -1.0, 0.0],      # 12: Middle Tip
        [0.1, -0.4, 0.0],      # 13: Ring MCP
        [0.12, -0.6, 0.0],     # 14
        [0.13, -0.75, 0.0],    # 15
        [0.14, -0.88, 0.0],    # 16: Ring Tip
        [0.2, -0.35, 0.0],     # 17: Pinky MCP
        [0.24, -0.5, 0.0],     # 18
        [0.26, -0.62, 0.0],    # 19
        [0.28, -0.74, 0.0]     # 20: Pinky Tip
    ], dtype=np.float32)

    # Modifikasi pose sesuai semantik kelas
    if class_idx == 0:  # saya: telunjuk menunjuk ke dada
        pose[8] = [-0.05, -0.4, 0.3]
        pose[12] = [0.0, -0.3, 0.1]
        pose[16] = [0.05, -0.28, 0.1]
        pose[20] = [0.1, -0.25, 0.1]
    elif class_idx == 1:  # kamu: telunjuk lurus ke depan
        pose[8] = [0.0, -0.85, -0.4]
        pose[12] = [0.02, -0.35, -0.05]
        pose[16] = [0.06, -0.32, -0.05]
        pose[20] = [0.1, -0.3, -0.05]
    elif class_idx == 2:  # anda: telapak tangan terbuka sopan menghadap depan
        pose[:, 2] -= 0.15
    elif class_idx == 3:  # kami: tangan melengkung ke dada (inklusif/eksklusif)
        pose[8] = [-0.15, -0.5, 0.2]
        pose[12] = [-0.05, -0.55, 0.2]
    elif class_idx == 4:  # kita: gerakan melingkar merangkul
        pose[4] = [-0.25, -0.3, 0.1]
        pose[8] = [-0.05, -0.7, 0.1]
    elif class_idx == 5:  # dia: menunjuk ke arah samping
        pose[8] = [0.6, -0.5, 0.0]
        pose[12] = [0.1, -0.3, 0.0]
    elif class_idx == 6:  # mereka: sapuan menunjuk ke samping jamak
        pose[8] = [0.7, -0.4, -0.1]
        pose[12] = [0.65, -0.42, -0.1]
    elif class_idx == 7:  # makan: jari-jari menguncup di dekat mulut
        pose[4] = [-0.05, -0.6, 0.2]
        pose[8] = [-0.02, -0.65, 0.2]
        pose[12] = [0.0, -0.66, 0.2]
        pose[16] = [0.02, -0.64, 0.2]
        pose[20] = [0.04, -0.6, 0.2]
    elif class_idx == 8:  # minum: ibu jari tegak membentuk cangkir
        pose[4] = [-0.1, -0.6, 0.1]
        pose[8:21] = pose[8:21] * 0.5
    elif class_idx == 9:  # tidur: telapak tangan miring menyangga pipi
        pose[:, 0] += 0.3
        pose[:, 1] *= 0.8
    elif class_idx == 10: # belajar: telapak tangan mendatar membuka buku
        pose[:, 1] = pose[:, 1] * 0.4
        pose[:, 2] += 0.2
    elif class_idx == 11: # bekerja: kedua kepalan saling ketuk
        pose[4:21] = pose[4:21] * 0.45
    elif class_idx == 12: # berjalan: dua jari (telunjuk & tengah) melangkah ke bawah
        pose[8] = [-0.1, 0.2, 0.0]
        pose[12] = [0.1, 0.15, 0.0]
        pose[4] = [-0.15, -0.2, 0.0]
        pose[16:21] = pose[16:21] * 0.3
    elif class_idx == 13: # membaca: telapak kiri meja, jari kanan menelusuri
        pose[8] = [0.1, -0.5, 0.1]
        pose[12] = [0.15, -0.52, 0.1]

    return pose

def augment_pose(pose):
    """Augmentasi spasial: rotasi acak, skala acak, dan gaussian jitter."""
    # 1. Jitter
    jitter = np.random.normal(0, 0.015, pose.shape).astype(np.float32)
    aug = pose + jitter

    # 2. Skala acak (0.85 - 1.15)
    scale = np.random.uniform(0.85, 1.15)
    aug = aug * scale

    # 3. Rotasi acak z-axis (rotasi planar)
    angle = np.random.uniform(-0.25, 0.25)  # radians (~ -15 to +15 deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ], dtype=np.float32)
    aug = np.dot(aug, rot_matrix.T)

    # 4. Normalisasi wrist-centered & scale
    wrist = aug[0].copy()
    translated = aug - wrist
    dist = np.linalg.norm(translated[12])
    if dist < 1e-6:
        dist = np.max(np.linalg.norm(translated, axis=1))
        if dist < 1e-6:
            dist = 1.0
    normalized = (translated / dist).flatten()
    return normalized.astype(np.float32)

def generate_sequence(class_idx, seq_len=12):
    """Menghasilkan urutan 12 frame dengan dinamika temporal."""
    base_pose = get_base_hand_pose(class_idx)
    frames = []
    
    # Kecepatan transisi dan drift temporal
    t_offsets = np.linspace(0, 1, seq_len)
    drift_dir = np.random.uniform(-0.05, 0.05, (3,)).astype(np.float32)

    for t in t_offsets:
        cur_pose = base_pose.copy()
        # Tambahkan dinamika gerak halus
        cur_pose += (drift_dir * t)
        norm_frame = augment_pose(cur_pose)
        frames.append(norm_frame)

    return np.array(frames, dtype=np.float32) # (12, 63)

def build_dataset(samples_per_class=150, seq_len=12):
    labels = load_labels()
    print(f"[DATASET] Membangun dataset untuk {len(labels)} kelas...")
    
    all_spatial_X = []
    all_spatial_y = []
    all_temporal_X = []
    all_temporal_y = []

    for c_idx, label in enumerate(labels):
        for _ in range(samples_per_class):
            seq = generate_sequence(c_idx, seq_len=seq_len)
            all_temporal_X.append(seq)
            all_temporal_y.append(c_idx)

            # Setiap frame dari sekuens juga menjadi sampel data spasial 1D-CNN
            for frame in seq:
                all_spatial_X.append(frame)
                all_spatial_y.append(c_idx)

    spatial_X = np.array(all_spatial_X, dtype=np.float32)
    spatial_y = np.array(all_spatial_y, dtype=np.int64)
    temporal_X = np.array(all_temporal_X, dtype=np.float32)
    temporal_y = np.array(all_temporal_y, dtype=np.int64)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, "dataset_landmarks.npz")
    np.savez_compressed(
        out_file,
        spatial_X=spatial_X,
        spatial_y=spatial_y,
        temporal_X=temporal_X,
        temporal_y=temporal_y,
        classes=np.array(labels)
    )
    print(f"[DATASET] Sukses disimpan di {out_file}!")
    print(f"  - Sampel Spasial 1D-CNN : {spatial_X.shape}, Labels: {spatial_y.shape}")
    print(f"  - Sekuens Temporal LSTM : {temporal_X.shape}, Labels: {temporal_y.shape}")

if __name__ == "__main__":
    build_dataset()
