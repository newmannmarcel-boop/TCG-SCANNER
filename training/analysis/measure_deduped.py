"""
Misst die Genauigkeit NACHDEM man die API-Duplikate (gleiche riftbound_id)
zusammenfasst UND set-filtert. Zeigt was ohne jegliches Training erreichbar ist.
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

cards, page = [], 1
while True:
    d = fetch(f"{API}/cards?size=100&page={page}")
    items = d.get("items", [])
    if not items: break
    cards.extend(items)
    if page >= d.get("pages", 1): break
    page += 1
cards = [c for c in cards if c.get("media", {}).get("image_url") and c.get("id")]

# entries in Trainingsreihenfolge (Klassenindex = i)
entries = []
for c in cards:
    p = CACHE / f"{c['id']}.png"
    if p.exists():
        entries.append((c["id"], (c.get("set") or {}).get("set_id","?"),
                        c.get("riftbound_id",""), p))

ck = torch.load(MODEL, map_location=DEVICE, weights_only=True)
n_cls = ck["num_classes"]
m = mobilenet_v3_small(weights=None)
inf = m.classifier[-1].in_features
m.classifier[-1] = nn.Linear(inf, n_cls)
m.load_state_dict(ck["model_state"]); m.eval().to(DEVICE)

random.seed(42); np.random.seed(42); torch.manual_seed(42)
val_tfm = phone_aug.build_phone_val_pipeline(IMG, T, torch)

set_of_idx = [s for _, s, _, _ in entries]
rbid_of_idx = [r for _, _, r, _ in entries]
idx_by_set = defaultdict(list)
for i, s in enumerate(set_of_idx):
    idx_by_set[s].append(i)

# Vorberechnete Vorhersagen (set-gefiltert)
print("Forward-Pass ...")
pred_setfiltered = [None]*len(entries)
with torch.no_grad():
    for i, (_, s, r, path) in enumerate(entries):
        x = val_tfm(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
        logits = m(x)[0]
        members = idx_by_set[s]
        pred_setfiltered[i] = members[int(torch.argmax(logits[members]))]

N = len(entries)
# A) Set-gefiltert, exakte Karte
exact = sum(1 for i in range(N) if pred_setfiltered[i] == i)

# B) Set-gefiltert, ABER Vorhersage zaehlt korrekt wenn gleiche riftbound_id
#    (= API-Duplikat/Reprint richtig erkannt, nur anderer DB-Eintrag)
rbid_ok = sum(1 for i in range(N) if rbid_of_idx[i] and rbid_of_idx[pred_setfiltered[i]] == rbid_of_idx[i])

print("\n" + "="*55)
print(f"Set-gefiltert, exakte DB-Klasse:            {exact/N*100:.1f}%")
print(f"Set-gefiltert + riftbound_id-Duplikate ok:  {rbid_ok/N*100:.1f}%")
print("="*55)
print("(letzteres = was erreichbar ist wenn API-Duplikate zusammengefasst werden)")

# Per-Set mit riftbound_id-Toleranz
print("\nPer-Set (set-gefiltert + Duplikate zusammengefasst):")
for s in sorted(idx_by_set):
    members = idx_by_set[s]
    ok = sum(1 for i in members if rbid_of_idx[i] and rbid_of_idx[pred_setfiltered[i]] == rbid_of_idx[i])
    print(f"  {s:6s}: {ok}/{len(members)} = {ok/len(members)*100:.1f}%")
