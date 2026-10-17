"""
Zeigt KONKRET welche VEN-Varianten sich eine Bild-URL teilen und WARUM:
volle Details (Name, Nummer, riftbound_id, Rarity, Metadata-Flags, URL).
"""
import re, time
from collections import defaultdict
import requests

API = "https://api.riftcodex.com"

def fetch(url, tries=6):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=30); r.raise_for_status(); return r.json()
        except Exception:
            if i == tries-1: raise
            time.sleep(2**i)

def base_name(name):
    return re.sub(r"\s*\((Signature|Alternate Art|Overnumbered|Alt Art|Promo)\)\s*", "", name, flags=re.I).strip().lower()

cards, page = [], 1
while True:
    d = fetch(f"{API}/cards?size=100&page={page}")
    items = d.get("items", [])
    if not items: break
    cards.extend(items)
    if page >= d.get("pages", 1): break
    page += 1

ven = [c for c in cards if (c.get("set") or {}).get("set_id") == "VEN"]
groups = defaultdict(list)
for c in ven:
    groups[base_name(c.get("name",""))].append(c)

def url_base(u): return (u or "").split("?")[0].split("/")[-1]  # nur Dateiname

# Kategorien zaehlen: welche Variantentypen teilen sich das Bild?
same_url_by_type = defaultdict(int)
diff_url_by_type = defaultdict(int)

def vtype(c):
    m = c.get("metadata", {})
    if m.get("signature"): return "Signature"
    if m.get("overnumbered"): return "Overnumbered"
    if m.get("alternate_art"): return "Alternate Art"
    return "Basis"

shown = 0
print("KONKRETE BEISPIELE (Basis vs Variante, gleiche Gruppe):\n")
for b, g in groups.items():
    if len(g) < 2: continue
    g = sorted(g, key=lambda c: c.get("collector_number", 0))
    ref = g[0]
    ref_file = url_base(ref["media"]["image_url"])
    for c in g[1:]:
        cf = url_base(c["media"]["image_url"])
        same = (cf == ref_file)
        if same: same_url_by_type[vtype(c)] += 1
        else:    diff_url_by_type[vtype(c)] += 1
        if shown < 8:
            print(f"'{ref.get('name')}' (Nr {ref.get('collector_number')}, {ref.get('riftbound_id')}, {vtype(ref)})")
            print(f"   vs '{c.get('name')}' (Nr {c.get('collector_number')}, {c.get('riftbound_id')}, {vtype(c)})")
            print(f"   Bild gleich? {'JA - identische Datei' if same else 'nein - anderes Bild'}  ({cf[:24]}...)")
            print()
            shown += 1

print("="*55)
print("Bild GETEILT (gleiche Datei) nach Variantentyp:")
for t, n in sorted(same_url_by_type.items(), key=lambda x:-x[1]):
    print(f"  {t:15s}: {n}")
print("\nBild VERSCHIEDEN nach Variantentyp:")
for t, n in sorted(diff_url_by_type.items(), key=lambda x:-x[1]):
    print(f"  {t:15s}: {n}")
