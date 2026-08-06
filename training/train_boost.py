"""
Boost-Runde: zusaetzlicher LR-Warm-Restart auf dem bereits Phone-Realistic
feingetunten Modell (best_model.pth), gleiche phone_aug-Pipeline.

Hintergrund: Das alte 1064-Klassen-Modell durchlief 3 LR-Schedules
(initial -> resume -> phone), das neue 1451-Klassen-Modell bisher nur 2
(train.py -> train_phone.py) und plateaut niedriger (~81% statt 89% val_phone).
Ein frischer Cosine-Restart mit moderater Lernrate kann aus so einem Plateau
rausfuehren (SGDR-Warm-Restart-Prinzip) -- ohne die Augmentation zu aendern.

Ueberschreibt best_model.pth NUR wenn val_phone den bisherigen Bestwert
tatsaechlich uebertrifft (kein Rueckschritt moeglich).
Resumable Checkpoint nach jeder Epoch (Modell+Optimizer+Scheduler).
"""
import os, json, time, random
from pathlib import Path
from io import BytesIO

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from torchvision.models import mobilenet_v3_large
from PIL import Image
import requests
from tqdm import tqdm

import phone_aug

try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _GPU_HANDLE = None

TARGET_POWER_W = 320  # Prozess-Drosselung: haelt NUR das Training zurueck,
                       # nicht die GPU als Ganzes (siehe throttle_gpu_power)


def throttle_gpu_power():
    """Bremst den Trainings-Takt nur, falls die GPU ueber TARGET_POWER_W zieht.
    Bei TARGET_POWER_W nahe dem Hardware-Maximum greift das praktisch nie ->
    Training laeuft mit voller Geschwindigkeit."""
    if _GPU_HANDLE is None:
        return
    while pynvml.nvmlDeviceGetPowerUsage(_GPU_HANDLE) / 1000.0 > TARGET_POWER_W:
        time.sleep(0.05)

SCRIPT_DIR = Path(__file__).parent
CACHE_DIR = SCRIPT_DIR / "card_cache"
OUTPUT_DIR = SCRIPT_DIR / "output"
API_BASE = "https://api.riftcodex.com"

IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS = 25
LEARNING_RATE = 1.8e-4  # moderater Warm-Restart (zwischen 2.5e-4 Start und ~0 Ende)
WEIGHT_DECAY = 1e-4
REPEATS_PER_EPOCH = 40
CHECKPOINT_EVERY = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_PATH = OUTPUT_DIR / "best_model.pth"
LAST_PATH = OUTPUT_DIR / "boost_resume.pth"


def fetch_card_list():
    print("[1/5] Karten-Liste laden ...")
    cards = []
    page = 1
    while True:
        r = fetch_json_with_retry(f"{API_BASE}/cards?size=100&page={page}")
        items = r.get("items", [])
        if not items:
            break
        cards.extend(items)
        tp = r.get("pages", 1)
        if page >= tp:
            break
        page += 1
    return [c for c in cards if c.get("media", {}).get("image_url") and c.get("id")]


def fetch_json_with_retry(url, attempts=6):
    """Riftcodex-API ist gelegentlich kurz nicht erreichbar (Timeout/Reset).
    Retry mit steigender Wartezeit, statt den ganzen Lauf abbrechen zu lassen."""
    for i in range(attempts):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if i == attempts - 1:
                raise
            wait = min(2 ** i, 30)
            print(f"  Netzwerkfehler ({e}), retry in {wait}s ...")
            time.sleep(wait)


def ensure_images(cards):
    print(f"[2/5] {len(cards)} Karten-Bilder (Cache) ...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for card in tqdm(cards, desc="Check"):
        p = CACHE_DIR / f"{card['id']}.png"
        if not p.exists():
            ok = False
            for attempt in range(4):
                try:
                    r = requests.get(card["media"]["image_url"], timeout=30)
                    r.raise_for_status()
                    Image.open(BytesIO(r.content)).convert("RGB").save(p)
                    ok = True
                    break
                except Exception as e:
                    if attempt == 3:
                        print(f"  Skip {card.get('name','?')}: {e}")
                    else:
                        time.sleep(min(2 ** attempt, 15))
            if not ok:
                continue
        paths.append((card["id"], card.get("name", "?"), p))
    return paths


class CardDataset(Dataset):
    def __init__(self, entries, transform):
        self.entries, self.transform = entries, transform
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, idx):
        _, _, path = self.entries[idx]
        return self.transform(Image.open(path).convert("RGB")), idx


class Repeated(Dataset):
    def __init__(self, base, n):
        self.base, self.n = base, n
    def __len__(self):
        return len(self.base) * self.n
    def __getitem__(self, idx):
        return self.base[idx % len(self.base)]


def load_model(num_classes):
    print(f"[3/5] Modell aus {BEST_PATH.name} laden ...")
    ck = torch.load(BEST_PATH, map_location=DEVICE, weights_only=True)
    start_val_phone = ck.get("val_phone", 0.0)
    print(f"  saved val_acc={ck['val_acc']*100:.2f}%  val_phone={start_val_phone*100:.2f}%")
    m = mobilenet_v3_large(weights=None)
    inf = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(inf, num_classes)
    m.load_state_dict(ck["model_state"])
    return m.to(DEVICE), start_val_phone


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return correct / max(1, total)


def train(model, train_loader, clean_loader, phone_loader, num_classes, start_best_phone):
    print(f"[4/5] Boost-Training auf {DEVICE} ({NUM_EPOCHS} Epochs, LR-Warm-Restart) ...")
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_phone = start_best_phone
    start_ep = 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if LAST_PATH.exists():
        ck = torch.load(LAST_PATH, map_location=DEVICE, weights_only=True)
        if ck.get("num_classes") == num_classes and "optimizer_state" in ck:
            model.load_state_dict(ck["model_state"])
            opt.load_state_dict(ck["optimizer_state"])
            sched.load_state_dict(ck["scheduler_state"])
            start_ep = ck["epoch"] + 1
            best_phone = ck["best_phone"]
            print(f"  -> Unterbrochener Boost-Lauf gefunden, Fortsetzung ab Epoch {start_ep+1}")
        else:
            print(f"  -> Resume-Checkpoint passt nicht, starte neu.")

    for ep in range(start_ep, NUM_EPOCHS):
        model.train()
        t0 = time.time()
        loss_sum = corr = tot = 0
        pbar = tqdm(train_loader, desc=f"Boost-Epoch {ep+1}/{NUM_EPOCHS}", leave=False)
        for step, (imgs, labels) in enumerate(pbar):
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            out = model(imgs)
            loss = crit(out, labels)
            loss.backward()
            opt.step()
            throttle_gpu_power()
            loss_sum += loss.item() * imgs.size(0)
            corr += (out.argmax(1) == labels).sum().item()
            tot += imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{corr/tot*100:.1f}%")

        train_acc = corr / max(1, tot)
        val_clean = evaluate(model, clean_loader)
        val_phone = evaluate(model, phone_loader)
        sched.step()
        # Gibt von PyTorch zwischengespeicherten (aber ungenutzten) VRAM an
        # Windows zurueck -> Browser/Video-Decoder haben wieder Speicher frei.
        torch.cuda.empty_cache()
        dt = time.time() - t0

        mark = ""
        if val_phone > best_phone:
            best_phone = val_phone
            torch.save({"model_state": model.state_dict(),
                        "num_classes": num_classes, "val_acc": val_clean,
                        "val_phone": val_phone}, BEST_PATH)
            mark = " <- neuer best (phone)"
        # Resumable Checkpoint NACH JEDER Epoch -> bei Absturz/Stromausfall
        # geht hoechstens die laufende Epoch verloren, nicht der ganze Lauf.
        if (ep + 1) % CHECKPOINT_EVERY == 0:
            torch.save({"model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict(),
                        "scheduler_state": sched.state_dict(),
                        "epoch": ep,
                        "num_classes": num_classes, "val_acc": val_clean,
                        "val_phone": val_phone, "best_phone": best_phone}, LAST_PATH)

        print(f"Boost-Epoch {ep+1:>2}/{NUM_EPOCHS}  loss={loss_sum/max(1,tot):.3f}  "
              f"train_acc={train_acc*100:.1f}%  val_clean={val_clean*100:.1f}%  "
              f"val_phone={val_phone*100:.1f}%  lr={sched.get_last_lr()[0]:.2e}  ({dt:.0f}s){mark}")

    if LAST_PATH.exists():
        LAST_PATH.unlink()
    print(f"-> Best val_phone (gesamt): {best_phone*100:.2f}%")


def export(num_classes, entries):
    print("[5/5] ONNX exportieren ...")
    ck = torch.load(BEST_PATH, map_location=DEVICE, weights_only=True)
    m = mobilenet_v3_large(weights=None)
    inf = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(inf, num_classes)
    m.load_state_dict(ck["model_state"])
    m.eval().to(DEVICE)
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    onnx_path = OUTPUT_DIR / "riftbound-model.onnx"
    torch.onnx.export(m, dummy, str(onnx_path), input_names=["input"],
                      output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=14)
    print(f"  -> {onnx_path} ({onnx_path.stat().st_size/1024/1024:.1f} MB)")
    meta = {"version": 1, "model": "mobilenet_v3_large_phone_finetune_boost",
            "input_size": IMAGE_SIZE, "num_classes": num_classes,
            "val_acc": ck.get("val_acc"), "val_phone": ck.get("val_phone"),
            "preprocess": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "cardIds": [c for (c, _n, _p) in entries],
            "cardNames": [n for (_c, n, _p) in entries]}
    with open(OUTPUT_DIR / "riftbound-model-meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print("  -> meta.json")


def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    cards = fetch_card_list()
    entries = ensure_images(cards)
    print(f"  {len(entries)} Karten\n")

    train_tfm = phone_aug.build_phone_pipeline(IMAGE_SIZE, T, torch)
    phone_val_tfm = phone_aug.build_phone_val_pipeline(IMAGE_SIZE, T, torch)
    clean_val_tfm = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_ds = Repeated(CardDataset(entries, train_tfm), REPEATS_PER_EPOCH)
    clean_ds = CardDataset(entries, clean_val_tfm)
    phone_ds = CardDataset(entries, phone_val_tfm)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=12, pin_memory=True,
                              persistent_workers=True, prefetch_factor=4)
    clean_loader = DataLoader(clean_ds, batch_size=BATCH_SIZE, num_workers=4,
                              pin_memory=True, persistent_workers=True)
    phone_loader = DataLoader(phone_ds, batch_size=BATCH_SIZE, num_workers=4,
                              pin_memory=True, persistent_workers=True)

    model, start_val_phone = load_model(len(entries))
    train(model, train_loader, clean_loader, phone_loader, len(entries), start_val_phone)
    export(len(entries), entries)
    print("\n[OK] Fertig.")


if __name__ == "__main__":
    main()
