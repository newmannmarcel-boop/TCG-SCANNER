"""
Prueft ob die Riftcodex-API echte Duplikate liefert (dieselbe Karte mehrfach),
die im Training als getrennte Klassen mit identischem Bild landen ->
per Konstruktion unklassifizierbar, ziehen die Genauigkeit kuenstlich runter.
"""
import time
from collections import Counter, defaultdict
import requests

API = "https://api.riftcodex.com"

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

# Nur die mit Bild+id (wie im Training)
valid = [c for c in cards if c.get("media", {}).get("image_url") and c.get("id")]
print(f"API liefert {len(cards)} Karten, davon {len(valid)} mit Bild+id\n")

# 1) Doppelte MongoDB-ids?
id_counts = Counter(c["id"] for c in valid)
dupe_ids = {i: n for i, n in id_counts.items() if n > 1}
print(f"Doppelte 'id' (exakt selbe DB-id mehrfach): {len(dupe_ids)}")

# 2) Doppelte riftbound_id?
rb_counts = Counter(c.get("riftbound_id") for c in valid)
dupe_rb = {i: n for i, n in rb_counts.items() if n > 1 and i}
print(f"Doppelte 'riftbound_id': {len(dupe_rb)} verschiedene ids, "
      f"betrifft {sum(rb_counts[i] for i in dupe_rb)} Karten")

# 3) Gleiche Bild-URL global (nicht nur VEN)?
def url_file(u): return (u or "").split("?")[0].split("/")[-1]
img_counts = Counter(url_file(c["media"]["image_url"]) for c in valid)
dupe_img = {i: n for i, n in img_counts.items() if n > 1}
print(f"Gleiche Bild-Datei global: {len(dupe_img)} Bilder werden von "
      f"{sum(img_counts[i] for i in dupe_img)} Karten geteilt")

# 4) Wieviele EINDEUTIGE Bilder gibt es ueberhaupt?
print(f"\n-> {len(img_counts)} eindeutige Bilder fuer {len(valid)} Karten-Klassen")
print(f"   d.h. {len(valid)-len(img_counts)} Klassen haben KEIN eindeutiges Bild")

# 5) Beispiele: gleiche riftbound_id, verschiedene DB-id
print("\nBeispiele doppelte riftbound_id (verschiedene DB-Eintraege, selbe Karte):")
by_rb = defaultdict(list)
for c in valid:
    by_rb[c.get("riftbound_id")].append(c)
shown = 0
for rb, g in by_rb.items():
    if len(g) > 1 and rb:
        names = [c.get("name") for c in g]
        imgs = [url_file(c["media"]["image_url"]) for c in g]
        all_same_img = len(set(imgs)) == 1
        print(f"  {rb}: {len(g)}x | Namen={set(names)} | Bild identisch={all_same_img}")
        shown += 1
        if shown >= 12: break
