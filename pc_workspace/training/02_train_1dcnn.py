"""
02_train_1dcnn.py
Pelatihan Model 1D-CNN untuk Ekstraksi Fitur Spasial Landmark (63-d).
Menghasilkan bobot ONNX: data/models/spatial_1dcnn.onnx
Output ONNX:
  1. class_logits: (batch, 14) - Probabilitas klasifikasi instan
  2. spatial_features: (batch, 64) - Vektor representasi untuk input LSTM
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
OUTPUT_MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
ONNX_OUTPUT_PATH = os.path.join(OUTPUT_MODEL_DIR, "spatial_1dcnn.onnx")

class Spatial1DCNN(nn.Module):
    def __init__(self, in_features=63, num_classes=14):
        super().__init__()
        # Input: (batch, 1, 63)
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2) # 63 // 2 = 31
        )
        self.fc_embed = nn.Sequential(
            nn.Linear(64 * 31, 64),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, 63) -> reshape to (batch, 1, 63)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.conv_block(x)
        feat = feat.view(feat.size(0), -1)
        embed = self.fc_embed(feat)
        logits = self.classifier(embed)
        return logits, embed

def train():
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset {DATASET_PATH} belum ada. Jalankan 01_prepare_dataset.py terlebih dahulu!")
        return

    data = np.load(DATASET_PATH)
    X = data["spatial_X"] # (N, 63)
    y = data["spatial_y"] # (N,)
    classes = data["classes"]
    num_classes = len(classes)

    print(f"[INFO] Memuat dataset spasial: {X.shape[0]} sampel, {num_classes} kelas.")

    # Split train/val (80/20)
    indices = np.random.permutation(len(X))
    split_idx = int(0.8 * len(X))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    X_train, y_train = torch.tensor(X[train_idx]), torch.tensor(y[train_idx])
    X_val, y_val = torch.tensor(X[val_idx]), torch.tensor(y[val_idx])

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Pelatihan berjalan di device: {device}")

    model = Spatial1DCNN(in_features=63, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    epochs = 10
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
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
                logits, _ = model(batch_x)
                preds = logits.argmax(dim=1)
                v_correct += (preds == batch_y).sum().item()
                v_total += batch_x.size(0)
        val_acc = v_correct / v_total

        print(f"Epoch [{epoch:2d}/{epochs:2d}] Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    # Export to ONNX
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    model.eval()
    dummy_input = torch.randn(1, 63, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_OUTPUT_PATH,
        input_names=["landmarks_63"],
        output_names=["class_logits", "spatial_features"],
        dynamic_axes={
            "landmarks_63": {0: "batch_size"},
            "class_logits": {0: "batch_size"},
            "spatial_features": {0: "batch_size"}
        },
        opset_version=18,
        dynamo=False
    )
    print(f"\n[SUKSES] Model 1D-CNN berhasil diekspor ke ONNX: {ONNX_OUTPUT_PATH}")

if __name__ == "__main__":
    train()
