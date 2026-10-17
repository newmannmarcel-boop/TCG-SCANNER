"""
Prueft die Hypothese: Vendetta's schlechte Rate kommt von Varianten mit
IDENTISCHEM Artwork (Alt-Art/Overnumbered/Signature), die aus einem einzigen
Studio-Bild physikalisch nicht unterscheidbar sind.

Vergleicht fuer VEN:
  A) exakte Karten-Genauigkeit (wie gehabt)
  B) Namens-gruppierte Genauigkeit: zaehlt als korrekt, wenn die vorhergesagte
     Karte DENSELBEN Basisnamen hat (= gleiches Artwork, nur andere Variante)
Springt B nahe 100%, ist es rein ein Varianten-Problem, kein Trainingsproblem.
"""
import sys, random, time, re
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from torchvision.models import mobilenet_v3_small
from PIL import Image
import requests

sys.path.insert(0, r"C:\Users\newma\Desktop\TCG-SCANNER\training")
import phone_aug

DEVICE = torch.device("cpu")
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

def base_name(name):
    # "Vi - Piltover Enforcer (Signature)" -> "vi - piltover enforcer"
    n = re.sub(r"\s*\((Signature|Alternate Art|Overnumbered|Alt Art|Promo)\)\s*", "", name, flags=re.I)
    return n.strip().lower()

cards, page = [], 1
while True:
    d = fetch(f"{API}/cards?size=100&page={page}")
    items = d.get("items", [])
    if not items: break
    cards.extend(items)
    if page >= d.get("pages", 1): break
    page += 1
cards = [c for c in cards if c.get("media", {}).get("image_url") and c.get("id")]
entries = []
for c in cards:
    p = CACHE / f"{c['id']}.png"
    if p.exists():
        set_id = (c.get("set") or {}).get("set_id", "?")
        entries.append((c["id"], set_id, c.get("name","?"), p))

ck = torch.load(MODEL, map_location=DEVICE, weights_only=True)
n_cls = ck["num_classes"]
m = mobilenet_v3_small(weights=None)
inf = m.classifier[-1].in_features
m.classifier[-1] = nn.Linear(inf, n_cls)
m.load_state_dict(ck["model_state"]); m.eval().to(DEVICE)

random.seed(42); np.random.seed(42); torch.manual_seed(42)
val_tfm = phone_aug.build_phone_val_pipeline(IMG, T, torch)

names = [n for _, _, n, _ in entries]
bases = [base_name(n) for n in names]
set_of_idx = [s for _, s, _, _ in entries]
idx_by_set = defaultdict(list)
for i, s in enumerate(set_of_idx):
    idx_by_set[s].append(i)

# Varianten-Statistik VEN
ven_idx = idx_by_set["VEN"]
ven_base_groups = defaultdict(list)
for i in ven_idx:
    ven_base_groups[bases[i]].append(i)
multi = {b: g for b, g in ven_base_groups.items() if len(g) > 1}
print(f"VEN: {len(ven_idx)} Karten, {len(ven_base_groups)} eindeutige Artworks, "
      f"{len(multi)} Artworks mit mehreren Varianten "
      f"({sum(len(g) for g in multi.values())} Karten in Varianten-Gruppen)")

print("\nForward-Pass nur VEN (set-gefiltert) ...")
exact = grouped = 0
with torch.no_grad():
    for i in ven_idx:
        _, s, name, path = entries[i]
        x = val_tfm(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
        logits = m(x)[0]
        members = idx_by_set[s]
        pred = members[int(torch.argmax(logits[members]))]
        if pred == i:
            exact += 1
        if bases[pred] == bases[i]:   # gleiches Artwork -> Varianten-Verwechslung ist "ok"
            grouped += 1

N = len(ven_idx)
print("\n" + "="*55)
print(f"VEN exakte Karten-Genauigkeit (set-gefiltert): {exact/N*100:.1f}%")
print(f"VEN Namens-gruppiert (Artwork richtig erkannt): {grouped/N*100:.1f}%")
print("="*55)
print(f"\n-> {grouped-exact} der Fehler sind reine Varianten-Verwechslungen")
print(f"   (gleiches Artwork, nur falsche Variante -> per OCR/Filter loesbar)")
