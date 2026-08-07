"""
Exportiert das archivierte Small-Modell (11,5 MB, gleiche Genauigkeit wie Large,
aber viel leichter fuers Handy) nach ONNX.

SICHERHEIT: Die Klassen-Reihenfolge MUSS exakt der des Modells entsprechen, sonst
gibt es falsche Karten-Zuordnungen. Wir bauen die Reihenfolge via fetch_card_list
neu auf UND vergleichen die cardIds mit der bereits deployten v4-Meta (Large).
Sind sie identisch, ist die Reihenfolge bewiesen korrekt -> Export. Sonst Abbruch.
"""
import json, time
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
import requests

SCRIPT_DIR = Path(__file__).parent
ARCHIVE = SCRIPT_DIR / "archive" / "best_model_small_1451classes_81pct.pth"
V4_META = SCRIPT_DIR.parent / "riftbound-model-meta-v4.json"   # Large-Meta (Referenz-Reihenfolge)
OUT_ONNX = SCRIPT_DIR.parent / "riftbound-model-v5.onnx"
OUT_META = SCRIPT_DIR.parent / "riftbound-model-meta-v5.json"
API = "https://api.riftcodex.com"
IMG = 224
DEVICE = torch.device("cpu")

def fetch(url, tries=6):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=30); r.raise_for_status(); return r.json()
        except Exception:
            if i == tries-1: raise
            time.sleep(2**i)

print("[1/4] Karten-Reihenfolge via fetch_card_list neu aufbauen ...")
cards, page = [], 1
while True:
    d = fetch(f"{API}/cards?size=100&page={page}")
    items = d.get("items", [])
    if not items: break
    cards.extend(items)
    if page >= d.get("pages", 1): break
    page += 1
entries = [c for c in cards if c.get("media", {}).get("image_url") and c.get("id")]
card_ids = [c["id"] for c in entries]
card_names = [c.get("name", "?") for c in entries]
print(f"  {len(card_ids)} Karten")

print("[2/4] Reihenfolge gegen deployte v4-Meta verifizieren ...")
v4 = json.load(open(V4_META, encoding="utf-8"))
if card_ids == v4["cardIds"]:
    print("  OK: Reihenfolge IDENTISCH zur v4-Meta -> Small-Modell passt garantiert.")
else:
    # Wie viele weichen ab?
    same = sum(1 for a, b in zip(card_ids, v4["cardIds"]) if a == b)
    print(f"  ABBRUCH: Reihenfolge weicht ab ({same}/{len(card_ids)} gleich, "
          f"Laengen {len(card_ids)} vs {len(v4['cardIds'])}).")
    raise SystemExit("Reihenfolge nicht identisch -> Export abgebrochen (Sicherheit).")

print("[3/4] Small-Modell laden + nach ONNX exportieren ...")
ck = torch.load(ARCHIVE, map_location=DEVICE, weights_only=True)
n = ck["num_classes"]
assert n == len(card_ids), f"Klassenzahl {n} != Kartenzahl {len(card_ids)}"
m = mobilenet_v3_small(weights=None)
inf = m.classifier[-1].in_features
m.classifier[-1] = nn.Linear(inf, n)
m.load_state_dict(ck["model_state"])
m.eval()
dummy = torch.randn(1, 3, IMG, IMG)
torch.onnx.export(m, dummy, str(OUT_ONNX), input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=14)
print(f"  -> {OUT_ONNX.name} ({OUT_ONNX.stat().st_size/1024/1024:.1f} MB)")

print("[4/4] Meta schreiben (gleiche cardIds/Reihenfolge) ...")
meta = {"version": 1, "model": "mobilenet_v3_small_phone_finetune",
        "input_size": IMG, "num_classes": n,
        "val_acc": ck.get("val_acc"), "val_phone": ck.get("val_phone"),
        "preprocess": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "cardIds": card_ids, "cardNames": card_names}
json.dump(meta, open(OUT_META, "w", encoding="utf-8"), ensure_ascii=False)
print(f"  -> {OUT_META.name}")
print("\n[OK] Small v5 exportiert. val_acc",
      round(ck.get('val_acc',0)*100,2), "% val_phone", round(ck.get('val_phone',0)*100,2), "%")
