"""
Fortsetzungs-Training: lädt best_model.pth und trainiert weiter.
Nutzt niedrigere Lernrate (Fine-Tuning) und robuste Checkpoints
alle 5 Epochs (falls Session abbricht wie beim letzten Mal).

Vom Ursprungs-Setup abweichend:
  - LR halbiert (2.5e-4 statt 5e-4) — schont bestehende Weights
  - Höhere Repeats-Anzahl (30 statt 20) für mehr Augmentations-Vielfalt
  - Stärkere Augmentation (mehr Rotation, mehr Glare, mehr Erasing)
  - Checkpoint alle 5 Epochs (letztes_checkpoint.pth zusätzlich zu best)
  - 40 zusätzliche Epochs oben drauf
"""
import os
import sys
import json
import time
import random
from pathlib import Path
from io import BytesIO

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from torchvision.models import mobilenet_v3_small
from PIL import Image, ImageFilter
import requests
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
CACHE_DIR = SCRIPT_DIR / "card_cache"
OUTPUT_DIR = SCRIPT_DIR / "output"
API_BASE = "https://api.riftcodex.com"

IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS_ADDITIONAL = 40  # ADDITIONAL Epochs (nach den ersten 6)
LEARNING_RATE = 2.5e-4      # Halbiert für Fine-Tuning
WEIGHT_DECAY = 1e-4
REPEATS_PER_EPOCH = 30      # Erhöht für mehr Aug-Vielfalt
CHECKPOINT_EVERY = 5        # alle N Epochs sichern

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_PATH = OUTPUT_DIR / "best_model.pth"
LAST_PATH = OUTPUT_DIR / "last_checkpoint.pth"


def fetch_card_list():
    print("[1/5] Karten-Liste laden ...")
    cards = []
    page = 1
    while True:
        r = requests.get(f"{API_BASE}/cards?size=100&page={page}", timeout=30).json()
        items = r.get("items", [])
        if not items:
            break
        cards.extend(items)
        total_pages = r.get("pages", 1)
        print(f"  Seite {page}/{total_pages}")
        if page >= total_pages:
            break
        page += 1
    return [c for c in cards if c.get("media", {}).get("image_url") and c.get("id")]


def download_card_images(cards):
    print(f"[2/5] {len(cards)} Karten-Bilder (Cache) ...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for card in tqdm(cards, desc="Check"):
        cache_path = CACHE_DIR / f"{card['id']}.png"
        if not cache_path.exists():
            try:
                r = requests.get(card["media"]["image_url"], timeout=30)
                img = Image.open(BytesIO(r.content)).convert("RGB")
                img.save(cache_path)
            except Exception as e:
                print(f"  Skip {card.get('name','?')}: {e}")
                continue
        paths.append((card["id"], card.get("name","?"), cache_path))
    return paths


class SyntheticGlare:
    """Foil-Glare Simulation mit weichen Gauss-Blur Spots."""
    def __init__(self, p=0.6, n_spots=(1, 4), spot_radius=(20, 80), intensity=(140, 230)):
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
            tr = random.randint(220, 255)
            tg = random.randint(210, 255)
            tb = random.randint(180, 255)
            overlay = Image.new("RGB", (r*2, r*2), (tr, tg, tb))
            mask = Image.new("L", (r*2, r*2), 0)
            ImageDraw.Draw(mask).ellipse((0, 0, r*2, r*2), fill=intens)
            mask = mask.filter(ImageFilter.GaussianBlur(radius=r/3))
            img.paste(overlay, (cx-r, cy-r), mask)
        return img


def build_transforms():
    """Aggressivere Augmentation gegenüber ursprünglichem Setup."""
    train_tfm = T.Compose([
        T.Resize((int(IMAGE_SIZE * 1.18), int(IMAGE_SIZE * 1.18))),
        T.RandomCrop(IMAGE_SIZE),
        T.RandomRotation(degrees=18, fill=128),               # von 15 auf 18
        T.RandomPerspective(distortion_scale=0.22, p=0.5, fill=128),  # von 0.18/0.4
        T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.3, hue=0.06),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.8))], p=0.4),
        SyntheticGlare(p=0.6),                                # von 0.5
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),          # von 0.2/0.12
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
    def __init__(self, card_paths, transform):
        self.entries = card_paths
        self.transform = transform
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, idx):
        _, _, path = self.entries[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), idx


class RepeatedCardDataset(Dataset):
    def __init__(self, base_ds, n_repeats):
        self.base = base_ds
        self.n_repeats = n_repeats
    def __len__(self):
        return len(self.base) * self.n_repeats
    def __getitem__(self, idx):
        return self.base[idx % len(self.base)]


def build_model_from_checkpoint(num_classes):
    print(f"[3/5] Modell aus {BEST_PATH.name} laden ...")
    if not BEST_PATH.exists():
        raise FileNotFoundError(f"Kein bestehendes Modell in {BEST_PATH}")
    ckpt = torch.load(BEST_PATH, map_location=DEVICE, weights_only=True)
    saved_classes = ckpt["num_classes"]
    saved_val = ckpt["val_acc"]
    print(f"  saved: num_classes={saved_classes}, val_acc={saved_val*100:.2f}%")
    if saved_classes != num_classes:
        raise ValueError(f"Class-Count Mismatch: {saved_classes} vs {num_classes}")
    model = mobilenet_v3_small(weights=None)
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, num_classes)
    model.load_state_dict(ckpt["model_state"])
    return model.to(DEVICE), saved_val


def train_loop(model, train_loader, val_loader, num_classes, start_best_val):
    print(f"[4/5] Fine-Tune-Training auf {DEVICE} ({NUM_EPOCHS_ADDITIONAL} Epochs) ...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_ADDITIONAL)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_acc = start_best_val
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(NUM_EPOCHS_ADDITIONAL):
        model.train()
        t0 = time.time()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS_ADDITIONAL} train", leave=False)
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

        model.eval()
        val_correct, val_total = 0, 0
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
                        "val_acc": val_acc}, BEST_PATH)
            marker = " <- neuer best"

        # Periodischer Checkpoint UNABHÄNGIG von val_acc
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            torch.save({"model_state": model.state_dict(),
                        "num_classes": num_classes,
                        "val_acc": val_acc}, LAST_PATH)

        print(f"Epoch {epoch+1:>2}/{NUM_EPOCHS_ADDITIONAL}  train_loss={train_loss:.3f}  "
              f"train_acc={train_acc*100:.1f}%  val_acc={val_acc*100:.1f}%  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  ({dt:.1f}s){marker}")

    print(f"-> Best Val-Acc: {best_val_acc*100:.2f}%")


def export_onnx(num_classes, card_paths):
    print("[5/5] ONNX exportieren ...")
    ckpt = torch.load(BEST_PATH, map_location=DEVICE, weights_only=True)
    model = mobilenet_v3_small(weights=None)
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(DEVICE)
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    onnx_path = OUTPUT_DIR / "riftbound-model.onnx"
    torch.onnx.export(model, dummy, str(onnx_path),
                       input_names=["input"], output_names=["logits"],
                       dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                       opset_version=14)
    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"  -> {onnx_path} ({size_mb:.1f} MB)")

    meta = {
        "version": 1,
        "model": "mobilenet_v3_small_finetune_resumed",
        "input_size": IMAGE_SIZE,
        "num_classes": num_classes,
        "val_acc": ckpt["val_acc"],
        "preprocess": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "cardIds":   [cid for (cid, _n, _p) in card_paths],
        "cardNames": [name for (_c, name, _p) in card_paths],
    }
    meta_path = OUTPUT_DIR / "riftbound-model-meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"  -> {meta_path}")


def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    cards = fetch_card_list()
    card_paths = download_card_images(cards)
    print(f"  {len(card_paths)} Karten verfuegbar\n")

    train_tfm, val_tfm = build_transforms()
    base_train_ds = CardDataset(card_paths, transform=train_tfm)
    val_ds = CardDataset(card_paths, transform=val_tfm)
    train_ds = RepeatedCardDataset(base_train_ds, n_repeats=REPEATS_PER_EPOCH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=12, pin_memory=True, persistent_workers=True,
                              prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True,
                            prefetch_factor=2)

    model, start_best_val = build_model_from_checkpoint(num_classes=len(card_paths))
    train_loop(model, train_loader, val_loader, len(card_paths), start_best_val)
    export_onnx(len(card_paths), card_paths)
    print("\n[OK] Fertig.")


if __name__ == "__main__":
    main()
