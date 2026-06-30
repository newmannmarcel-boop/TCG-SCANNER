"""
Riftbound Card Classifier — Custom Training

Trainiert MobileNetV3-Small auf allen Riftbound-Karten von Riftcodex.
Heavy Augmentation simuliert echte Handy-Foto-Bedingungen.
Output: ONNX-Modell (im Browser via onnxruntime-web nutzbar).

Workflow:
  1) Karten-Liste + Bilder von Riftcodex laden (cached lokal)
  2) Augmentation-Pipeline aufsetzen
  3) MobileNetV3-Small fine-tunen
  4) Best-Model speichern + zu ONNX exportieren
"""

import os
import sys
import json
import time
import math
import random
import urllib.request
from pathlib import Path
from io import BytesIO

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image, ImageFilter
import requests
from tqdm import tqdm

# ===== KONFIG =====
SCRIPT_DIR = Path(__file__).parent
CACHE_DIR = SCRIPT_DIR / "card_cache"
OUTPUT_DIR = SCRIPT_DIR / "output"
API_BASE = "https://api.riftcodex.com"

IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS = 40
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
VALIDATION_SPLIT = 0.1  # 10% pro Klasse zur Validation
AUG_PER_EPOCH = 1       # eine Augmentation pro Karte pro Epoch (Online-Augmentation)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 1) KARTEN-DATEN LADEN =====

def fetch_card_list():
    """Holt alle Karten von Riftcodex API (paginiert)."""
    print("[1/5] Karten-Liste laden ...")
    cards = []
    page = 1
    while True:
        resp = requests.get(f"{API_BASE}/cards?size=100&page={page}", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        if not items:
            break
        cards.extend(items)
        total_pages = data.get("pages", 1)
        print(f"  Seite {page}/{total_pages} -> {len(cards)} Karten")
        if page >= total_pages:
            break
        page += 1
    return [c for c in cards if c.get("media", {}).get("image_url") and c.get("id")]


def download_card_images(cards):
    """Lädt jedes Karten-Bild und cached lokal."""
    print(f"[2/5] {len(cards)} Karten-Bilder laden ...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    skipped_idx = []
    for i, card in enumerate(tqdm(cards, desc="Download")):
        cache_path = CACHE_DIR / f"{card['id']}.png"
        if not cache_path.exists():
            try:
                url = card["media"]["image_url"]
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                # Konvertiere zu RGB-PNG (für Konsistenz)
                img = Image.open(BytesIO(r.content)).convert("RGB")
                img.save(cache_path)
            except Exception as e:
                print(f"  Skip {card.get('name','?')}: {e}")
                skipped_idx.append(i)
                continue
        paths.append((card["id"], card.get("name", "?"), cache_path))
    if skipped_idx:
        print(f"  Übersprungen: {len(skipped_idx)} Karten")
    return paths


# ===== 2) DATASET MIT AUGMENTATION =====

class SyntheticGlare:
    """Fügt zufällige weich-abgrenzte helle Flecken hinzu (Foil-Reflexionen).
    Volle Qualität: Gauss-Blurred-Mask für realistischen Glanz-Look."""
    def __init__(self, p=0.5, n_spots=(1, 3), spot_radius=(20, 70), intensity=(120, 220)):
        self.p, self.n_spots, self.spot_radius, self.intensity = p, n_spots, spot_radius, intensity
    def __call__(self, img):
        if random.random() > self.p:
            return img
        img = img.copy()
        w, h = img.size
        from PIL import ImageDraw
        for _ in range(random.randint(*self.n_spots)):
            cx, cy = random.randint(0, w), random.randint(0, h)
            r = random.randint(*self.spot_radius)
            intens = random.randint(*self.intensity)
            # Leicht goldener/silbriger Tint zufällig (echte Foils sind nicht reinweiß)
            tint_r = random.randint(220, 255)
            tint_g = random.randint(210, 255)
            tint_b = random.randint(180, 255)
            overlay = Image.new("RGB", (r*2, r*2), (tint_r, tint_g, tint_b))
            mask = Image.new("L", (r*2, r*2), 0)
            ImageDraw.Draw(mask).ellipse((0, 0, r*2, r*2), fill=intens)
            mask = mask.filter(ImageFilter.GaussianBlur(radius=r/3))
            img.paste(overlay, (cx - r, cy - r), mask)
        return img


def build_transforms():
    """Augmentation für robustes Real-World-Training.
    Volle Qualität: Perspektive, Gauss-Blur, Glare — alles drin damit das
    Modell auch auf wackligen Handy-Fotos mit Foil-Glanz klarkommt."""
    train_tfm = T.Compose([
        T.Resize((int(IMAGE_SIZE * 1.15), int(IMAGE_SIZE * 1.15))),
        T.RandomCrop(IMAGE_SIZE),
        T.RandomRotation(degrees=15, fill=128),
        T.RandomPerspective(distortion_scale=0.18, p=0.4, fill=128),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.5))], p=0.35),
        SyntheticGlare(p=0.5),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.RandomErasing(p=0.2, scale=(0.02, 0.12)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tfm = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tfm, val_tfm


class CardDataset(Dataset):
    """Eine Klasse pro Karte. Augmentation passiert online beim Aufruf."""
    def __init__(self, card_paths, transform):
        # card_paths: [(card_id, name, path), ...]
        self.entries = card_paths
        self.transform = transform
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, idx):
        _, _, path = self.entries[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), idx


class RepeatedCardDataset(Dataset):
    """Erweiterte Dataset-Variante: jedes Bild wird n_repeats mal pro Epoch
    gezogen, jedes Mal mit neuer Augmentation. Mehr Trainings-Iterationen
    ohne Image-Re-Read."""
    def __init__(self, base_ds, n_repeats):
        self.base = base_ds
        self.n_repeats = n_repeats
    def __len__(self):
        return len(self.base) * self.n_repeats
    def __getitem__(self, idx):
        return self.base[idx % len(self.base)]


# ===== 3) MODELL =====

def build_model(num_classes):
    print(f"[3/5] MobileNetV3-Small aufsetzen, {num_classes} Klassen ...")
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # Final classifier durch unsere Anzahl Klassen ersetzen
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, num_classes)
    return model.to(DEVICE)


# ===== 4) TRAINING =====

def train_model(model, train_loader, val_loader, num_classes):
    print(f"[4/5] Training auf {DEVICE} ({NUM_EPOCHS} Epochs) ...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_acc = 0.0
    best_path = OUTPUT_DIR / "best_model.pth"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        # ----- TRAIN -----
        model.train()
        t0 = time.time()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} train", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{train_correct/train_total*100:.1f}%")

        train_loss = train_loss_sum / max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        # ----- VAL -----
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                logits = model(imgs)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
        val_acc = val_correct / max(1, val_total)

        scheduler.step()
        dt = time.time() - t0

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(),
                        "num_classes": num_classes,
                        "val_acc": val_acc}, best_path)
            marker = " ← neuer best"

        print(f"Epoch {epoch+1:>2}/{NUM_EPOCHS}  train_loss={train_loss:.3f}  "
              f"train_acc={train_acc*100:.1f}%  val_acc={val_acc*100:.1f}%  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  ({dt:.1f}s){marker}")

    print(f"-> Best Val-Acc: {best_val_acc*100:.2f}%, Modell: {best_path}")
    return best_path


# ===== 5) ONNX EXPORT =====

def export_onnx(checkpoint_path, card_paths):
    print("[5/5] ONNX exportieren ...")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model = mobilenet_v3_small(weights=None)
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(DEVICE)

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    onnx_path = OUTPUT_DIR / "riftbound-model.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14,
    )
    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"-> {onnx_path} ({size_mb:.1f} MB)")

    # Metadaten: cardIds in der korrekten Reihenfolge (Index -> cardId)
    meta_path = OUTPUT_DIR / "riftbound-model-meta.json"
    meta = {
        "version": 1,
        "model": "mobilenet_v3_small_finetune",
        "input_size": IMAGE_SIZE,
        "num_classes": ckpt["num_classes"],
        "val_acc": ckpt["val_acc"],
        "preprocess": {
            "mean": [0.485, 0.456, 0.406],
            "std":  [0.229, 0.224, 0.225],
        },
        "cardIds": [cid for (cid, _name, _path) in card_paths],
        "cardNames": [name for (_cid, name, _path) in card_paths],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"-> {meta_path}")
    return onnx_path, meta_path


# ===== MAIN =====

def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    cards = fetch_card_list()
    card_paths = download_card_images(cards)
    print(f"\n{len(card_paths)} Karten verfügbar für Training\n")

    train_tfm, val_tfm = build_transforms()
    base_train_ds = CardDataset(card_paths, transform=train_tfm)
    val_ds = CardDataset(card_paths, transform=val_tfm)
    # Über-Sampling per Epoch: 20× durch jede Karte mit randomized augmentation
    # → 1064 × 20 = ~21k Samples/Epoch, bei 40 Epochs ~850k augmentierte Bilder
    train_ds = RepeatedCardDataset(base_train_ds, n_repeats=20)

    # 12 Worker für 20-Core-CPU + hoher prefetch_factor → GPU bleibt füttert
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=12, pin_memory=True, persistent_workers=True,
                              prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True,
                            prefetch_factor=2)

    model = build_model(num_classes=len(card_paths))
    best_path = train_model(model, train_loader, val_loader, len(card_paths))
    export_onnx(best_path, card_paths)
    print("\n[OK] Fertig. ONNX-Modell im training/output/ Verzeichnis.")


if __name__ == "__main__":
    main()
