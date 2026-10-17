"""
Zerlegt die Karten mit GETEILTEM Bild danach, WARUM sie sich ein Bild teilen.
Beantwortet: kommen die ~280 wirklich von echten Duplikaten, oder ist die
Messung irrefuehrend?
"""
import re, time
from collections import defaultdict, Counter
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
valid = [c for c in cards if c.get("media", {}).get("image_url") and c.get("id")]

def url_file(u): return (u or "").split("?")[0].split("/")[-1]

# Gruppiere nach Bild-Datei
by_img = defaultdict(list)
for c in valid:
    by_img[url_file(c["media"]["image_url"])].append(c)

shared = {img: g for img, g in by_img.items() if len(g) > 1}
print(f"{len(shared)} Bilder werden von mehreren Karten geteilt "
      f"({sum(len(g) for g in shared.values())} Karten betroffen)\n")

# Kategorisiere jede geteilte Gruppe
cat = Counter()
extra_classes = Counter()  # wieviele "ueberzaehlige" Klassen pro Kategorie
examples = defaultdict(list)

def has_suffix(name, kw):
    return kw.lower() in name.lower()

for img, g in shared.items():
    names = [c.get("name","") for c in g]
    rbs = [c.get("riftbound_id","") for c in g]
    sets = set((c.get("set") or {}).get("set_id") for c in g)
    extra = len(g) - 1  # eine davon ist "das echte" Bild, der Rest ist Zusatz

    # Kategorie bestimmen
    if len(set(rbs)) == 1:
        # ALLE haben dieselbe riftbound_id -> echtes DB-Duplikat derselben Karte
        c = "1_echtes_DB_duplikat_gleiche_rbid"
    elif any(has_suffix(n, "metal") for n in names):
        c = "2_metal_variante"
    elif any(has_suffix(n, "overnumber") for n in names):
        c = "3_overnumbered"
    elif any(has_suffix(n, "alternate") or has_suffix(n, "alt art") for n in names):
        c = "4_alternate_art_gleiches_bild"
    elif any(has_suffix(n, "promo") for n in names):
        c = "5_promo"
    elif len(sets) > 1:
        c = "6_cross_set_gleiches_bild"
    else:
        c = "7_sonstiges"
    cat[c] += 1
    extra_classes[c] += extra
    if len(examples[c]) < 6:
        examples[c].append((names, rbs, list(sets)))

print("KATEGORIEN (Anzahl geteilter Bild-Gruppen | ueberzaehlige Klassen):")
for c in sorted(cat):
    print(f"  {c}: {cat[c]} Gruppen | +{extra_classes[c]} Extra-Klassen")
print(f"\n  SUMME Extra-Klassen (= 'Duplikate'): {sum(extra_classes.values())}")

print("\n\nBEISPIELE pro Kategorie:")
for c in sorted(examples):
    print(f"\n--- {c} ---")
    for names, rbs, sets in examples[c]:
        print(f"  Namen={names}")
        print(f"    rbids={rbs} sets={sets}")
