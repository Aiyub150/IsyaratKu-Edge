"""
03_train_lstm.py
Pelatihan Model LSTM Temporal untuk Sekuens Fitur Gestur (12 frame x 64 dimensi).
Menerima representasi fitur dari 1D-CNN dan mengklasifikasikan ke 14 kelas.
Menghasilkan bobot ONNX: data/models/temporal_lstm.onnx
"""

import os
import sys
import io

# Pastikan UTF-8 encoding untuk Windows terminal
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(BASE_DIR, "data", "landmarks_cache", "dataset_landmarks.npz")
CNN_MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "spatial_1dcnn.onnx")
OUTPUT_MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
ONNX_OUTPUT_PATH = os.path.join(OUTPUT_MODEL_DIR, "temporal_lstm.onnx")

class TemporalLSTM(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, num_layers=2, num_classes=14):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len=12, input_dim=64)
        lstm_out, _ = self.lstm(x)
        # Ambil state frame terakhir
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return out

def encode_with_1dcnn(temporal_X):
    """
    Melewatkan data landmark (N, 12, 63) melalui 1D-CNN ONNX untuk menghasilkan
    vektor fitur spasial (N, 12, 64).
    """
    import onnxruntime as ort
    session = ort.InferenceSession(CNN_MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    # Cari nama output untuk spatial_features
    output_names = [o.name for o in session.get_outputs()]
    feat_output_name = "spatial_features" if "spatial_features" in output_names else output_names[-1]

    N, seq_len, _ = temporal_X.shape
    flat_X = temporal_X.reshape(-1, 63).astype(np.float32)
    outputs = session.run([feat_output_name], {input_name: flat_X})
    encoded_flat = outputs[0] # (N * 12, 64)
    encoded_seq = encoded_flat.reshape(N, seq_len, -1)
    return encoded_seq

def train():
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset {DATASET_PATH} belum ada. Jalankan 01_prepare_dataset.py terlebih dahulu!")
        return

    if not os.path.exists(CNN_MODEL_PATH):
        print(f"[ERROR] Model 1D-CNN {CNN_MODEL_PATH} belum ada. Jalankan 02_train_1dcnn.py terlebih dahulu!")
        return

    data = np.load(DATASET_PATH)
    raw_temporal_X = data["temporal_X"] # (N, 12, 63)
    y = data["temporal_y"] # (N,)
    classes = data["classes"]
    num_classes = len(classes)

    print(f"[INFO] Mengekstrak fitur 1D-CNN dari {raw_temporal_X.shape[0]} sekuens...")
    X = encode_with_1dcnn(raw_temporal_X)
    feature_dim = X.shape[-1]
    print(f"[INFO] Fitur spasial siap: shape {X.shape} (dimensi={feature_dim}).")

    # Split train/val (80/20)
    indices = np.random.permutation(len(X))
    split_idx = int(0.8 * len(X))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    X_train, y_train = torch.tensor(X[train_idx]), torch.tensor(y[train_idx])
    X_val, y_val = torch.tensor(X[val_idx]), torch.tensor(y[val_idx])

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Pelatihan berjalan di device: {device}")

    model = TemporalLSTM(input_dim=feature_dim, hidden_dim=64, num_layers=2, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    epochs = 12
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        train_acc = correct / total
        avg_loss = total_loss / total

        # Validation
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                preds = outputs.argmax(dim=1)
                v_correct += (preds == batch_y).sum().item()
                v_total += batch_x.size(0)
        val_acc = v_correct / v_total

        print(f"Epoch [{epoch:2d}/{epochs:2d}] Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    # Export to ONNX
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    model.eval()
    dummy_input = torch.randn(1, 12, feature_dim, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_OUTPUT_PATH,
        input_names=["sequence_12x64"],
        output_names=["class_logits"],
        dynamic_axes={"sequence_12x64": {0: "batch_size"}, "class_logits": {0: "batch_size"}},
        opset_version=18,
        dynamo=False
    )
    print(f"\n[SUKSES] Model LSTM Temporal berhasil diekspor ke ONNX: {ONNX_OUTPUT_PATH}")

if __name__ == "__main__":
    train()
