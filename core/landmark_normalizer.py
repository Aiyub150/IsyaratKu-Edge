import numpy as np

def normalize_landmarks(raw_landmarks):
    """
    Menormalisasi 21 koordinat landmark 3D MediaPipe agar invariant terhadap
    translasi posisi tangan di frame dan skala/jarak tangan ke kamera.

    Alur Normalisasi:
    1. Translasi: Kurangkan seluruh koordinat (x, y, z) dengan koordinat Wrist (Landmark 0).
       Hasilnya membuat koordinat Wrist menjadi (0, 0, 0).
    2. Skala: Bagi seluruh koordinat dengan jarak Euclidean maksimum antara Wrist (0)
       dan Ujung Jari Tengah (Middle Finger Tip / Landmark 12).
       Jika jarak mendekati 0 (tangan mengepal/error), gunakan epsilon 1e-6.

    Args:
        raw_landmarks: Bisa berupa:
            - list of dict: [{'x': ..., 'y': ..., 'z': ...}, ...] (21 elemen)
            - MediaPipe NormalizedLandmarkList
            - numpy array bentuk (21, 3) atau (63,)

    Returns:
        numpy.ndarray: Vektor 1D berukuran (63,) bertipe float32 dalam rentang [-1.0, 1.0].
    """
    coords = []
    
    # 1. Ekstrak koordinat menjadi numpy array (21, 3)
    if hasattr(raw_landmarks, "landmark"):
        # MediaPipe LandmarkList object
        for lm in raw_landmarks.landmark:
            coords.append([lm.x, lm.y, lm.z])
        coords = np.array(coords, dtype=np.float32)
    elif isinstance(raw_landmarks, (list, tuple)):
        if len(raw_landmarks) == 21:
            if isinstance(raw_landmarks[0], dict):
                for lm in raw_landmarks:
                    coords.append([lm.get('x', 0.0), lm.get('y', 0.0), lm.get('z', 0.0)])
                coords = np.array(coords, dtype=np.float32)
            elif hasattr(raw_landmarks[0], 'x'):
                for lm in raw_landmarks:
                    coords.append([lm.x, lm.y, lm.z])
                coords = np.array(coords, dtype=np.float32)
            elif isinstance(raw_landmarks[0], (list, tuple, np.ndarray)):
                coords = np.array(raw_landmarks, dtype=np.float32)
        elif len(raw_landmarks) == 63:
            coords = np.array(raw_landmarks, dtype=np.float32).reshape(21, 3)
    elif isinstance(raw_landmarks, np.ndarray):
        if raw_landmarks.shape == (21, 3):
            coords = raw_landmarks.astype(np.float32)
        elif raw_landmarks.shape == (63,):
            coords = raw_landmarks.reshape(21, 3).astype(np.float32)

    if len(coords) != 21:
        raise ValueError(f"Landmark harus memiliki 21 titik, diterima: {len(coords)}")

    # 2. Translasi terhadap Wrist (Landmark 0)
    wrist = coords[0].copy()
    translated = coords - wrist

    # 3. Skala: Hitung jarak Euclidean ke Middle Finger Tip (Landmark 12)
    # Landmark 12 = Ujung jari tengah
    scale_dist = np.linalg.norm(translated[12])
    if scale_dist < 1e-6:
        # Fallback jika jari tengah terlipat: gunakan jarak Euclidean maksimum dari titik manapun ke wrist
        scale_dist = np.max(np.linalg.norm(translated, axis=1))
        if scale_dist < 1e-6:
            scale_dist = 1.0

    normalized = translated / scale_dist

    # 4. Ratakan menjadi vektor 1D (63,)
    return normalized.flatten().astype(np.float32)
