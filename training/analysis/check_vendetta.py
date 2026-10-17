import json, urllib.request, sys

API = "https://api.riftcodex.com"

def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

def fetch_all_set(set_id):
    cards = []
    page = 1
    while True:
        url = f"{API}/cards?size=100&page={page}&set_id={set_id}"
        data = fetch_json(url)
        items = data.get("items", [])
        if not items:
            break
        cards.extend(items)
        if page >= data.get("pages", 1):
            break
        page += 1
    return cards

for set_id in ["VEN", "SFD", "OPP"]:
    cards = fetch_all_set(set_id)
    with_img = [c for c in cards if c.get("media", {}).get("image_url")]
    print(f"Set {set_id}: {len(cards)} Karten total, {len(with_img)} mit Bild-URL")

# Sample check: are Vendetta image URLs reachable (just check format, not actual HTTP HEAD to avoid rate limit issues)
ven = fetch_all_set("VEN")
print("\nBeispiel Vendetta Karten:")
for c in ven[:5]:
    print(" -", c.get("name"), "|", c.get("riftbound_id"), "|", c.get("classification", {}).get("rarity"), "|", c.get("media", {}).get("image_url", "")[:60])

# Check rarity distribution
from collections import Counter
rarities = Counter(c.get("classification", {}).get("rarity") for c in ven)
print("\nSeltenheits-Verteilung Vendetta:", dict(rarities))

d = fetch_json(f"{API}/cards?size=1&page=1")
print("\nGesamt-Kartenanzahl API (alle Sets):", d.get("total"))
