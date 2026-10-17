"""
Prueft ob Vendetta-Varianten mit gleichem Namen UNTERSCHIEDLICHE Bilder haben.
Entscheidet ob 288px-Training ueberhaupt helfen KANN:
  - Byte-identisch -> Aufloesung hilft NIE (unlernbar)
  - visuell fast identisch -> Aufloesung hilft kaum
  - deutlich verschieden -> 288px hat echte Chance
"""
import sys, re, hashlib, time
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import requests

CACHE = Path(r"C:\Users\newma\Desktop\TCG-SCANNER\training\card_cache")
API = "https://api.riftcodex.com"

def fetch(url, tries=6):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=30); r.raise_for_status(); return r.json()
        except Exception:
            if i == tries-1: raise
            time.sleep(2**i)

def base_name(name):
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

# Nur VEN, nur mit lokalem Bild
ven = []
for c in cards:
    if (c.get("set") or {}).get("set_id") != "VEN": continue
    if not c.get("id"): continue
    p = CACHE / f"{c['id']}.png"
    if p.exists():
        ven.append((c["id"], c.get("name","?"), c["media"]["image_url"], p))

groups = defaultdict(list)
for cid, name, url, p in ven:
    groups[base_name(name)].append((cid, name, url, p))
multi = {b: g for b, g in groups.items() if len(g) > 1}
print(f"VEN: {len(multi)} Artwork-Gruppen mit mehreren Varianten\n")

def file_hash(p):
    return hashlib.md5(p.read_bytes()).hexdigest()

def small_arr(p, size=64):
    return np.asarray(Image.open(p).convert("RGB").resize((size,size))).astype(np.float32)

def url_base(u):
    # CMS-URL ohne Query -> zeigt ob dieselbe Bilddatei referenziert wird
    return u.split("?")[0]

identical_file = 0
identical_url = 0
near = 0
distinct = 0
examples = []
for b, g in multi.items():
    # paarweise gegen das erste Mitglied
    ref_hash = file_hash(g[0][3])
    ref_url = url_base(g[0][2])
    ref_arr = small_arr(g[0][3])
    for cid, name, url, p in g[1:]:
        if url_base(url) == ref_url:
            identical_url += 1
            examples.append((name, g[0][1], "GLEICHE Bild-URL"))
        elif file_hash(p) == ref_hash:
            identical_file += 1
            examples.append((name, g[0][1], "byte-identische Datei"))
        else:
            diff = np.abs(small_arr(p) - ref_arr).mean()
            if diff < 3.0:
                near += 1
                examples.append((name, g[0][1], f"fast identisch (diff={diff:.1f})"))
            else:
                distinct += 1

total_pairs = identical_url + identical_file + near + distinct
print(f"Varianten-Paare gesamt: {total_pairs}")
print(f"  Gleiche Bild-URL (Riftcodex liefert dasselbe Bild): {identical_url}")
print(f"  Byte-identische Dateien:                            {identical_file}")
print(f"  Visuell fast identisch (diff<3):                    {near}")
print(f"  Deutlich verschieden (288px koennte helfen):        {distinct}")
print()
print("Beispiele identischer/fast-identischer Varianten:")
for name, ref, why in examples[:15]:
    print(f"  '{name}' vs '{ref}': {why}")
