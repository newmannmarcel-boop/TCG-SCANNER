"""
Exportiert das bereits trainierte best_model.pth als ONNX + Metadaten.
Kein Re-Training, nur Konvertierung.
"""
import os
import json
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
import requests

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"
API_BASE = "https://api.riftcodex.com"
IMAGE_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Karten-Liste holen (für Metadaten — Reihenfolge MUSS gleich sein wie beim Training)
print("[1/3] Karten-Liste holen ...")
cards = []
page = 1
while True:
    r = requests.get(f"{API_BASE}/cards?size=100&page={page}", timeout=30).json()
    items = r.get("items", [])
    if not items:
        break
    cards.extend(items)
    total_pages = r.get("pages", 1)
    if page >= total_pages:
        break
    page += 1
# Filter wie im Training: nur Karten mit Image
cards = [c for c in cards if c.get("media", {}).get("image_url") and c.get("id")]
print(f"  {len(cards)} Karten")

# Checkpoint laden
print("[2/3] Modell laden ...")
ckpt = torch.load(OUTPUT_DIR / "best_model.pth", map_location=DEVICE, weights_only=True)
num_classes = ckpt["num_classes"]
val_acc = ckpt["val_acc"]
print(f"  num_classes={num_classes}, val_acc={val_acc*100:.2f}%")
assert num_classes == len(cards), f"Class-Count Mismatch: {num_classes} vs {len(cards)}"

model = mobilenet_v3_small(weights=None)
in_feats = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_feats, num_classes)
model.load_state_dict(ckpt["model_state"])
model.eval().to(DEVICE)

# ONNX-Export
print("[3/3] ONNX exportieren ...")
dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
onnx_path = OUTPUT_DIR / "riftbound-model.onnx"
torch.onnx.export(
    model, dummy, str(onnx_path),
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=14,
)
size_mb = onnx_path.stat().st_size / 1024 / 1024
print(f"  -> {onnx_path} ({size_mb:.1f} MB)")

# Metadaten
meta = {
    "version": 1,
    "model": "mobilenet_v3_small_finetune",
    "input_size": IMAGE_SIZE,
    "num_classes": num_classes,
    "val_acc": val_acc,
    "epochs_trained": 6,
    "preprocess": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "cardIds":   [c["id"] for c in cards],
    "cardNames": [c.get("name", "?") for c in cards],
}
meta_path = OUTPUT_DIR / "riftbound-model-meta.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False)
print(f"  -> {meta_path}")
print("\n[OK] Fertig.")
