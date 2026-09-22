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

def get_base_hand_pose(class_idx, variant=0):
    """
    Menghasilkan pose dasar 21 landmark 3D (x, y, z) untuk masing-masing dari 14 kelas
    berdasarkan kamus gestur SIBI/BISINDO resmi dan umpan balik pengguna (Feedback #3).
    """
    # Base template: open palm facing camera
    pose = np.array([
        [0.0, 0.0, 0.0],       # 0: Wrist
        [-0.15, -0.1, -0.05],  # 1: Thumb CMC
        [-0.25, -0.2, -0.08],  # 2: Thumb MCP
        [-0.32, -0.3, -0.1],   # 3: Thumb IP
        [-0.38, -0.4, -0.12],  # 4: Thumb Tip
        [-0.1, -0.4, 0.0],     # 5: Index MCP
        [-0.12, -0.6, 0.0],    # 6: Index PIP
        [-0.13, -0.75, 0.0],   # 7: Index DIP
        [-0.14, -0.9, 0.0],    # 8: Index Tip
        [0.0, -0.42, 0.0],     # 9: Middle MCP
        [0.0, -0.65, 0.0],     # 10: Middle PIP
        [0.0, -0.82, 0.0],     # 11: Middle DIP
        [0.0, -1.0, 0.0],      # 12: Middle Tip
        [0.1, -0.4, 0.0],      # 13: Ring MCP
        [0.12, -0.6, 0.0],     # 14: Ring PIP
        [0.13, -0.75, 0.0],    # 15: Ring DIP
        [0.14, -0.88, 0.0],    # 16: Ring Tip
        [0.2, -0.35, 0.0],     # 17: Pinky MCP
        [0.24, -0.5, 0.0],     # 18: Pinky PIP
        [0.26, -0.62, 0.0],    # 19: Pinky DIP
        [0.28, -0.74, 0.0]     # 20: Pinky Tip
    ], dtype=np.float32)

    # 0. SAYA: SIBI Resmi (2 Variasi Didukung):
    #   Variasi A (variant % 2 == 0): 1 jari (telunjuk) menunjuk ke dada sendiri, jari lain mengepal
    #   Variasi B (variant % 2 == 1): Jempol & kelingking terbuka (Y-shape/Shaka), 3 jari lain mengepal, punggung tangan menghadap depan
    if class_idx == 0:
        if variant % 2 == 0:
            # Variasi A: Telunjuk menunjuk ke dada
            pose[5:9] = np.array([[-0.1, -0.4, 0.0], [-0.08, -0.5, 0.15], [-0.06, -0.58, 0.28], [-0.04, -0.62, 0.40]])
            pose[9:13] = np.array([[0.0, -0.42, 0.0], [0.0, -0.28, 0.06], [0.0, -0.20, 0.06], [0.0, -0.14, 0.05]])
            pose[13:17] = np.array([[0.1, -0.4, 0.0], [0.1, -0.26, 0.06], [0.1, -0.18, 0.06], [0.1, -0.12, 0.05]])
            pose[17:21] = np.array([[0.2, -0.35, 0.0], [0.2, -0.24, 0.06], [0.2, -0.16, 0.06], [0.2, -0.10, 0.05]])
            pose[1:5] = np.array([[-0.15, -0.1, -0.05], [-0.2, -0.2, 0.0], [-0.12, -0.25, 0.08], [-0.04, -0.26, 0.08]])
        else:
            # Variasi B: SIBI Y-sign (Jempol & Kelingking terbuka, telunjuk-tengah-manis mengepal, punggung tangan ke kamera)
            # Jempol terbuka keluar-atas
            pose[1:5] = np.array([[-0.15, -0.12, 0.0], [-0.28, -0.22, 0.0], [-0.38, -0.34, 0.0], [-0.48, -0.45, 0.0]])
            # 3 Jari tengah mengepal rapat ke telapak
            pose[5:9] = np.array([[-0.08, -0.38, 0.0], [-0.08, -0.22, 0.08], [-0.08, -0.16, 0.08], [-0.08, -0.10, 0.06]])
            pose[9:13] = np.array([[0.0, -0.40, 0.0], [0.0, -0.24, 0.08], [0.0, -0.17, 0.08], [0.0, -0.11, 0.06]])
            pose[13:17] = np.array([[0.08, -0.38, 0.0], [0.08, -0.22, 0.08], [0.08, -0.16, 0.08], [0.08, -0.10, 0.06]])
            # Kelingking terbuka tegak ke atas
            pose[17:21] = np.array([[0.18, -0.32, 0.0], [0.22, -0.50, 0.0], [0.25, -0.66, 0.0], [0.28, -0.80, 0.0]])
            # Punggung tangan menghadap ke depan (balik koordinat Z dan X)
            pose[:, 2] = -pose[:, 2]

    # 1. KAMU: SIBI Resmi (2 Variasi Didukung):
    #   Variasi A (variant % 2 == 0): 1 jari (telunjuk) menunjuk ke depan lawan bicara/kamera
    #   Variasi B (variant % 2 == 1): Telapak tangan terbuka 5 jari menghadap ke depan lawan bicara/kamera
    elif class_idx == 1:
        if variant % 2 == 0:
            # Variasi A: Telunjuk menunjuk lurus ke depan
            pose[5:9] = np.array([[-0.1, -0.4, 0.0], [-0.09, -0.52, -0.22], [-0.08, -0.60, -0.45], [-0.07, -0.66, -0.65]])
            pose[9:13] = np.array([[0.0, -0.42, 0.0], [0.0, -0.28, 0.05], [0.0, -0.20, 0.05], [0.0, -0.14, 0.04]])
            pose[13:17] = np.array([[0.1, -0.4, 0.0], [0.1, -0.26, 0.05], [0.1, -0.18, 0.05], [0.1, -0.12, 0.04]])
            pose[17:21] = np.array([[0.2, -0.35, 0.0], [0.2, -0.24, 0.05], [0.2, -0.16, 0.05], [0.2, -0.10, 0.04]])
            pose[1:5] = np.array([[-0.15, -0.1, -0.05], [-0.2, -0.2, 0.0], [-0.12, -0.25, 0.08], [-0.04, -0.26, 0.08]])
        else:
            # Variasi B: Telapak tangan terbuka 5 jari sopan menghadap lawan bicara (eks-anda)
            pose[:, 2] = 0.0

    # 2. ANDA: SIBI Resmi (Feedback #4 Poin 7):
    #   Mengepal tangan dengan bagian alas tangan / pergelangan menghadap ke depan seolah mengepal sesuatu.
    elif class_idx == 2:
        # Seluruh 4 jari mengepal rapat ke telapak
        pose[5:9] = np.array([[-0.08, -0.36, 0.05], [-0.08, -0.22, 0.12], [-0.08, -0.16, 0.12], [-0.08, -0.10, 0.09]])
        pose[9:13] = np.array([[0.0, -0.38, 0.05], [0.0, -0.24, 0.12], [0.0, -0.17, 0.12], [0.0, -0.11, 0.09]])
        pose[13:17] = np.array([[0.08, -0.36, 0.05], [0.08, -0.22, 0.12], [0.08, -0.16, 0.12], [0.08, -0.10, 0.09]])
        pose[17:21] = np.array([[0.15, -0.30, 0.05], [0.15, -0.20, 0.12], [0.15, -0.14, 0.12], [0.15, -0.09, 0.09]])
        # Ibu jari mengunci di atas kepalan jari
        pose[1:5] = np.array([[-0.12, -0.10, 0.0], [-0.15, -0.20, 0.06], [-0.06, -0.24, 0.14], [0.02, -0.24, 0.14]])
        # Rotasi agar alas kepalan (base of fist/wrist) menghadap ke depan kamera
        # Rotasi pitch ke depan (~50-65 derajat)
        pitch = 0.95 + (0.1 * (variant % 3 - 1))
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        for i in range(21):
            y, z = pose[i, 1], pose[i, 2]
            pose[i, 1] = y * cos_p - z * sin_p
            pose[i, 2] = y * sin_p + z * cos_p

    # 3. KAMI: Pose statis Huruf 'K' (SIBI Resmi) - 2 jari (telunjuk & tengah) tegak lurus, jari lain mengepal
    elif class_idx == 3:
        pose[5:9] = np.array([[-0.1, -0.4, 0.0], [-0.12, -0.6, 0.0], [-0.14, -0.75, 0.0], [-0.16, -0.90, 0.0]])
        pose[9:13] = np.array([[0.0, -0.42, 0.0], [0.05, -0.65, 0.0], [0.08, -0.82, 0.0], [0.10, -1.0, 0.0]])
        pose[13:17] = np.array([[0.1, -0.4, 0.0], [0.1, -0.26, 0.05], [0.1, -0.18, 0.05], [0.1, -0.12, 0.05]])
        pose[17:21] = np.array([[0.2, -0.35, 0.0], [0.2, -0.24, 0.05], [0.2, -0.16, 0.05], [0.2, -0.10, 0.05]])
        pose[1:5] = np.array([[-0.15, -0.1, -0.05], [-0.2, -0.2, 0.0], [-0.1, -0.25, 0.08], [0.02, -0.26, 0.08]])

    # 4. KITA: Pose statis Huruf 'W' (SIBI Resmi) - 3 jari (telunjuk, tengah, manis) tegak lurus, kelingking & ibu jari bertemu
    elif class_idx == 4:
        pose[5:9] = np.array([[-0.1, -0.4, 0.0], [-0.12, -0.6, 0.0], [-0.14, -0.75, 0.0], [-0.15, -0.90, 0.0]])
        pose[9:13] = np.array([[0.0, -0.42, 0.0], [0.0, -0.65, 0.0], [0.0, -0.82, 0.0], [0.0, -1.0, 0.0]])
        pose[13:17] = np.array([[0.1, -0.4, 0.0], [0.12, -0.6, 0.0], [0.14, -0.75, 0.0], [0.15, -0.88, 0.0]])
        pose[17:21] = np.array([[0.2, -0.35, 0.0], [0.2, -0.24, 0.05], [0.2, -0.16, 0.05], [0.2, -0.10, 0.05]])
        pose[1:5] = np.array([[-0.15, -0.1, -0.05], [-0.2, -0.2, 0.0], [-0.05, -0.25, 0.08], [0.1, -0.26, 0.08]])

    # 5. DIA: SIBI Resmi (Feedback #4 Poin 5):
    #   1 jari (telunjuk) menunjuk ke samping secara STATIS (orang ketiga tunggal), jari lain mengepal rapat
    elif class_idx == 5:
        pose[5:9] = np.array([[0.1, -0.4, 0.0], [0.28, -0.45, 0.0], [0.45, -0.48, 0.0], [0.62, -0.50, 0.0]])
        pose[9:13] = np.array([[0.0, -0.42, 0.0], [0.0, -0.26, 0.06], [0.0, -0.18, 0.06], [0.0, -0.12, 0.05]])
        pose[13:17] = np.array([[0.1, -0.4, 0.0], [0.1, -0.25, 0.06], [0.1, -0.17, 0.06], [0.1, -0.11, 0.05]])
        pose[17:21] = np.array([[0.2, -0.35, 0.0], [0.2, -0.23, 0.06], [0.2, -0.15, 0.06], [0.2, -0.09, 0.05]])
        pose[1:5] = np.array([[-0.15, -0.1, -0.05], [-0.2, -0.2, 0.0], [-0.12, -0.25, 0.08], [-0.04, -0.26, 0.08]])

    # 6. MEREKA: SIBI Resmi (Feedback #4 Poin 5):
    #   2 jari (telunjuk & tengah) menunjuk ke samping (orang ketiga jamak) dengan gerakan ayunan ke samping
    elif class_idx == 6:
        pose[5:9] = np.array([[0.1, -0.4, 0.0], [0.28, -0.45, 0.0], [0.45, -0.48, 0.0], [0.62, -0.50, 0.0]])
        pose[9:13] = np.array([[0.08, -0.38, 0.0], [0.26, -0.42, 0.0], [0.43, -0.45, 0.0], [0.60, -0.47, 0.0]])
        pose[13:17] = np.array([[0.1, -0.4, 0.0], [0.1, -0.25, 0.06], [0.1, -0.17, 0.06], [0.1, -0.11, 0.05]])
        pose[17:21] = np.array([[0.2, -0.35, 0.0], [0.2, -0.23, 0.06], [0.2, -0.15, 0.06], [0.2, -0.09, 0.05]])
        pose[1:5] = np.array([[-0.15, -0.1, -0.05], [-0.2, -0.2, 0.0], [-0.12, -0.25, 0.08], [-0.04, -0.26, 0.08]])

    # 7. MAKAN: Semua ujung jari menguncup bersamaan di dekat mulut
    elif class_idx == 7:
        pose[4] = [-0.03, -0.65, 0.2]
        pose[8] = [-0.01, -0.70, 0.2]
        pose[12] = [0.0, -0.71, 0.2]
        pose[16] = [0.01, -0.69, 0.2]
        pose[20] = [0.03, -0.65, 0.2]

    # 8. MINUM: SIBI Resmi (Feedback #4 Poin 4):
    #   Ibu jari dan 4 jari melengkung membentuk cangkir/gelas (C-shape) di depan bibir
    elif class_idx == 8:
        # C-shape: Jari telunjuk hingga kelingking melengkung memegang dinding gelas
        pose[5:9] = np.array([[-0.08, -0.35, 0.05], [-0.08, -0.45, 0.15], [-0.05, -0.52, 0.20], [-0.02, -0.54, 0.20]])
        pose[9:13] = np.array([[0.0, -0.36, 0.05], [0.0, -0.47, 0.16], [0.02, -0.54, 0.21], [0.04, -0.55, 0.21]])
        pose[13:17] = np.array([[0.08, -0.35, 0.05], [0.08, -0.45, 0.15], [0.09, -0.52, 0.20], [0.10, -0.53, 0.20]])
        pose[17:21] = np.array([[0.15, -0.32, 0.05], [0.15, -0.42, 0.14], [0.15, -0.48, 0.18], [0.15, -0.49, 0.18]])
        # Ibu jari berhadapan membentuk sisi lain cangkir/gelas
        pose[1:5] = np.array([[-0.14, -0.12, 0.0], [-0.22, -0.24, 0.06], [-0.20, -0.38, 0.12], [-0.14, -0.46, 0.18]])

    # 9. TIDUR: Telapak tangan miring menyangga pipi
    elif class_idx == 9:
        pose[:, 0] += 0.3
        pose[:, 1] *= 0.8

    # 10. BELAJAR: Telapak tangan mendatar membuka ke atas di depan dada (lembaran buku)
    elif class_idx == 10:
        pose[5:9] = np.array([[-0.1, -0.22, 0.0], [-0.1, -0.22, 0.15], [-0.1, -0.22, 0.30], [-0.1, -0.22, 0.45]])
        pose[9:13] = np.array([[0.0, -0.22, 0.0], [0.0, -0.22, 0.18], [0.0, -0.22, 0.35], [0.0, -0.22, 0.50]])
        pose[13:17] = np.array([[0.1, -0.22, 0.0], [0.1, -0.22, 0.16], [0.1, -0.22, 0.32], [0.1, -0.22, 0.46]])
        pose[17:21] = np.array([[0.2, -0.22, 0.0], [0.2, -0.22, 0.14], [0.2, -0.22, 0.28], [0.2, -0.22, 0.40]])
        pose[1:5] = np.array([[-0.15, -0.15, 0.0], [-0.25, -0.18, 0.05], [-0.32, -0.20, 0.1], [-0.38, -0.22, 0.15]])

    # 11. BEKERJA: Kedua kepalan saling mengetuk
    elif class_idx == 11:
        pose[4:21] = pose[4:21] * 0.45

    # 12. BERJALAN: Dua jari (telunjuk & tengah) melangkah ke bawah
    elif class_idx == 12:
        pose[8] = [-0.1, 0.2, 0.0]
        pose[12] = [0.1, 0.15, 0.0]
        pose[4] = [-0.15, -0.2, 0.0]
        pose[16:21] = pose[16:21] * 0.3

    # 13. MEMBACA: Telapak mendatar, telunjuk menelusuri baris bacaan
    elif class_idx == 13:
        pose[:, 1] = pose[:, 1] * 0.5
        pose[8] = [0.15, -0.45, 0.15]
        pose[12] = [0.2, -0.47, 0.15]

    return pose

def augment_pose(pose):
    """Augmentasi spasial: rotasi 3D (pitch, yaw, roll), skala acak, dan gaussian jitter."""
    # 1. Gaussian Jitter
    jitter = np.random.normal(0, 0.012, pose.shape).astype(np.float32)
    aug = pose + jitter

    # 2. Skala acak (0.85 - 1.15)
    scale = np.random.uniform(0.85, 1.15)
    aug = aug * scale

    # 3. Rotasi 3D acak:
    # a. Rotasi Roll (Z-axis) [-15, +15 deg]
    ang_z = np.random.uniform(-0.25, 0.25)
    cos_z, sin_z = np.cos(ang_z), np.sin(ang_z)
    rot_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]], dtype=np.float32)

    # b. Rotasi Pitch (X-axis) [-12, +12 deg]
    ang_x = np.random.uniform(-0.20, 0.20)
    cos_x, sin_x = np.cos(ang_x), np.sin(ang_x)
    rot_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]], dtype=np.float32)

    # c. Rotasi Yaw (Y-axis) [-12, +12 deg]
    ang_y = np.random.uniform(-0.20, 0.20)
    cos_y, sin_y = np.cos(ang_y), np.sin(ang_y)
    rot_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]], dtype=np.float32)

    rot_3d = np.dot(rot_z, np.dot(rot_y, rot_x))
    aug = np.dot(aug, rot_3d.T)

    # 4. Normalisasi wrist-centered & rigid palm scale (Wrist [0] ke Middle MCP [9])
    wrist = aug[0].copy()
    translated = aug - wrist
    dist = np.linalg.norm(translated[9])
    if dist < 1e-4:
        dist = np.max(np.linalg.norm(translated, axis=1))
        if dist < 1e-4:
            dist = 1.0
    normalized = (translated / dist).flatten()
    return normalized.astype(np.float32)

def generate_sequence(class_idx, seq_len=12, variant=0):
    """Menghasilkan urutan 12 frame dengan variasi dan dinamika temporal yang spesifik untuk setiap kelas."""
    base_pose = get_base_hand_pose(class_idx, variant=variant)
    frames = []
    
    t_offsets = np.linspace(0, 1, seq_len)

    # Dinamika temporal spesifik berdasarkan gestur SIBI:
    if class_idx == 5:
        # DIA: Statis (menunjuk samping tanpa gerakan sapuan)
        drift_dir = np.random.uniform(-0.01, 0.01, (3,)).astype(np.float32)
    elif class_idx == 6:
        # MEREKA: Ayunan sapuan horizontal (lateral sweep) sepanjang sumbu X
        sweep_dist = np.random.uniform(0.12, 0.25)
        drift_dir = np.array([sweep_dist, np.random.uniform(-0.02, 0.02), np.random.uniform(-0.02, 0.02)], dtype=np.float32)
    elif class_idx == 8:
        # MINUM: Gerakan mendekat ke mulut dan sedikit mendongak (tilting)
        drift_dir = np.array([np.random.uniform(-0.02, 0.02), np.random.uniform(-0.08, -0.03), np.random.uniform(0.04, 0.10)], dtype=np.float32)
    else:
        # Drift umum halus
        drift_dir = np.random.uniform(-0.04, 0.04, (3,)).astype(np.float32)

    for t in t_offsets:
        cur_pose = base_pose.copy()
        cur_pose += (drift_dir * t)
        norm_frame = augment_pose(cur_pose)
        frames.append(norm_frame)

    return np.array(frames, dtype=np.float32) # (12, 63)

def build_dataset(samples_per_class=250, seq_len=12):
    labels = load_labels()
    print(f"[DATASET] Membangun dataset terpadu SIBI Resmi (Riil Kamera + Sintetis) untuk {len(labels)} kelas...")
    
    raw_base_dir = os.path.join(BASE_DIR, "data", "dataset_raw")
    all_spatial_X = []
    all_spatial_y = []
    all_temporal_X = []
    all_temporal_y = []

    for c_idx, label in enumerate(labels):
        class_raw_dir = os.path.join(raw_base_dir, label)
        real_seqs = []

        # Kuota khusus: untuk 'dia' (5) dan 'mereka' (6), berikan sampel ekstra (450) agar terpisah tajam (Feedback #4 Poin 5)
        target_quota = 450 if label in ["dia", "mereka"] else samples_per_class

        # 1. Muat rekaman riil dari kamera jika ada
        if os.path.exists(class_raw_dir):
            npy_files = [f for f in os.listdir(class_raw_dir) if f.endswith(".npy")]
            for nf in npy_files:
                try:
                    arr = np.load(os.path.join(class_raw_dir, nf))
                    if arr.shape == (seq_len, 63):
                        real_seqs.append(arr.astype(np.float32))
                except Exception as e:
                    print(f"  [WARN] Gagal membaca {nf}: {e}")

        real_count = len(real_seqs)

        # 1. Selalu sertakan sekuens sintetis SIBI resmi berstandar Kamus SIBI
        print(f"  - [{label}]: Menghasilkan {target_quota} sekuens sintetis SIBI Resmi.")
        for s_idx in range(target_quota):
            seq = generate_sequence(c_idx, seq_len=seq_len, variant=s_idx)
            all_temporal_X.append(seq)
            all_temporal_y.append(c_idx)
            for frame in seq:
                all_spatial_X.append(frame)
                all_spatial_y.append(c_idx)

        # 2. Sertakan rekaman riil dari kamera jika ada untuk memperkaya variasi riil
        if real_count > 0:
            print(f"    + Menggabungkan {real_count} sekuens riil kamera (beserta augmentasi).")
            for seq in real_seqs:
                all_temporal_X.append(seq)
                all_temporal_y.append(c_idx)
                for frame in seq:
                    all_spatial_X.append(frame)
                    all_spatial_y.append(c_idx)

            # Tambahkan augmentasi dari sekuens riil
            aug_count = min(100, real_count * 5)
            for i in range(aug_count):
                src_seq = real_seqs[i % real_count]
                jitter = np.random.normal(0, 0.012, src_seq.shape).astype(np.float32)
                aug_seq = src_seq + jitter
                all_temporal_X.append(aug_seq)
                all_temporal_y.append(c_idx)
                for frame in aug_seq:
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
    print(f"\n[DATASET] Sukses disimpan di {out_file}!")
    print(f"  - Total Sampel Spasial 1D-CNN : {spatial_X.shape}, Labels: {spatial_y.shape}")
    print(f"  - Total Sekuens Temporal LSTM : {temporal_X.shape}, Labels: {temporal_y.shape}")

if __name__ == "__main__":
    build_dataset()
