"""
Misst den Effekt von Set-Filterung auf die Erkennungsgenauigkeit.
Laeuft READ-ONLY auf CPU -> stoert das laufende Large-Training nicht.

Vergleicht:
  - Global Top-1/3/5: argmax ueber ALLE 1451 Klassen
  - Per-Set Top-1: argmax nur ueber Karten des GLEICHEN Sets
Damit wird quantifiziert, wieviel die Set-Filter-Idee des Nutzers bringt.
"""
import sys, random, time
from pathlib import Path
from io import BytesIO
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from torchvision.models import mobilenet_v3_small
from PIL import Image
import requests

sys.path.insert(0, r"C:\Users\newma\Desktop\TCG-SCANNER\training")
import phone_aug

DEVICE = torch.device("cpu")  # bewusst CPU: GPU gehoert dem laufenden Training
CACHE = Path(r"C:\Users\newma\Desktop\TCG-SCANNER\training\card_cache")
MODEL = Path(r"C:\Users\newma\Desktop\TCG-SCANNER\training\archive\best_model_small_1451classes_81pct.pth")
API = "https://api.riftcodex.com"
IMG = 224

def fetch(url, tries=6):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=30); r.raise_for_status(); return r.json()
        except Exception:
            if i == tries-1: raise
            time.sleep(2**i)

print("Karten-Liste laden (gleiche Reihenfolge wie Training) ...")
cards, page = [], 1
while True:
    d = fetch(f"{API}/cards?size=100&page={page}")
    items = d.get("items", [])
    if not items: break
    cards.extend(items)
    if page >= d.get("pages", 1): break
    page += 1
# EXAKT gleiche Filterung wie fetch_card_list im Training:
cards = [c for c in cards if c.get("media", {}).get("image_url") and c.get("id")]
# Nur die, deren Bild lokal gecached ist (ensure_images Reihenfolge bleibt):
entries = []
for c in cards:
    p = CACHE / f"{c['id']}.png"
    if p.exists():
        set_id = (c.get("set") or {}).get("set_id", "?")
        entries.append((c["id"], set_id, p))
print(f"  {len(entries)} Karten mit lokalem Bild")

# Set-Verteilung
set_counts = Counter(s for _, s, _ in entries)
print("  Sets:", dict(set_counts))

# Modell laden
ck = torch.load(MODEL, map_location=DEVICE, weights_only=True)
n_cls = ck["num_classes"]
print(f"\nModell: {n_cls} Klassen, gespeichert val_phone={ck.get('val_phone',0)*100:.2f}%")
assert n_cls == len(entries), f"Klassenzahl {n_cls} != Kartenzahl {len(entries)} -> Reihenfolge stimmt nicht!"
m = mobilenet_v3_small(weights=None)
inf = m.classifier[-1].in_features
m.classifier[-1] = nn.Linear(inf, n_cls)
m.load_state_dict(ck["model_state"])
m.eval().to(DEVICE)

# Deterministische Phone-Val-Augmentation (geseedet fuer Reproduzierbarkeit)
random.seed(42); np.random.seed(42); torch.manual_seed(42)
val_tfm = phone_aug.build_phone_val_pipeline(IMG, T, torch)

# Set-Zugehoerigkeit pro Klassenindex + Masken vorbereiten
set_of_idx = [s for _, s, _ in entries]
idx_by_set = defaultdict(list)
for i, s in enumerate(set_of_idx):
    idx_by_set[s].append(i)

print("\nForward-Pass ueber alle Karten (CPU, dauert ~1-2 Min) ...")
top1 = top3 = top5 = 0
setfilter_correct = 0
per_set_correct = defaultdict(int)
t0 = time.time()
with torch.no_grad():
    for i, (_, s, path) in enumerate(entries):
        x = val_tfm(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
        logits = m(x)[0]
        # Global
        topk = torch.topk(logits, 5).indices.tolist()
        if i == topk[0]: top1 += 1
        if i in topk[:3]: top3 += 1
        if i in topk[:5]: top5 += 1
        # Set-gefiltert: nur Logits der Karten aus demselben Set betrachten
        members = idx_by_set[s]
        sub = logits[members]
        pred_local = members[int(torch.argmax(sub))]
        if pred_local == i:
            setfilter_correct += 1
            per_set_correct[s] += 1
        if (i+1) % 300 == 0:
            print(f"  {i+1}/{len(entries)} ({time.time()-t0:.0f}s)")

N = len(entries)
print("\n" + "="*55)
print(f"GLOBAL (argmax ueber alle {N} Klassen):")
print(f"  Top-1: {top1/N*100:.2f}%")
print(f"  Top-3: {top3/N*100:.2f}%")
print(f"  Top-5: {top5/N*100:.2f}%")
print(f"\nMIT SET-FILTER (argmax nur im jeweiligen Set):")
print(f"  Top-1: {setfilter_correct/N*100:.2f}%")
print(f"\nGewinn durch Set-Filter: +{(setfilter_correct-top1)/N*100:.2f} Prozentpunkte")
print("="*55)
print("\nPer-Set Top-1 (mit Set-Filter):")
for s in sorted(idx_by_set):
    n = len(idx_by_set[s])
    print(f"  {s:6s}: {per_set_correct[s]}/{n} = {per_set_correct[s]/n*100:.1f}%")
