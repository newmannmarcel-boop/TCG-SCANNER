// Berechnet MobileNet-v2-Embeddings für alle Riftbound-Karten
// und schreibt riftbound-embeddings.json ins Projekt-Root.

const tf = require('@tensorflow/tfjs');
const mobilenet = require('@tensorflow-models/mobilenet');
const { Jimp } = require('jimp');
const fs = require('fs');
const path = require('path');

const API_BASE = 'https://api.riftcodex.com';
const OUT_PATH = path.join(__dirname, '..', 'riftbound-embeddings.json');

async function fetchAllCards() {
  const cards = [];
  let page = 1, totalPages = 1;
  while (page <= totalPages) {
    const resp = await fetch(`${API_BASE}/cards?size=100&page=${page}`);
    if (!resp.ok) throw new Error(`API ${resp.status}`);
    const data = await resp.json();
    cards.push(...(data.items || []));
    totalPages = data.pages || 1;
    console.log(`  Seite ${page}/${totalPages} → ${cards.length} Karten`);
    page++;
  }
  return cards;
}

async function imageToTensor(url) {
  const img = await Jimp.read(url);
  // jimp lädt RGBA. Auf 224x224 resizen (was MobileNet erwartet)
  img.resize({ w: 224, h: 224 });
  const buffer = img.bitmap.data; // RGBA uint8, 224*224*4
  // RGB → uint8 Tensor [224, 224, 3] — der MobileNet-Wrapper macht Preprocessing selbst
  const rgb = new Uint8Array(224 * 224 * 3);
  for (let i = 0, j = 0; i < buffer.length; i += 4, j += 3) {
    rgb[j]     = buffer[i];
    rgb[j + 1] = buffer[i + 1];
    rgb[j + 2] = buffer[i + 2];
  }
  return tf.tensor3d(rgb, [224, 224, 3], 'int32');
}

async function main() {
  console.log('[1/3] Karten-Liste laden …');
  const cards = await fetchAllCards();
  const withImg = cards.filter(c => c.media && c.media.image_url && c.id);
  console.log(`→ ${cards.length} Karten, ${withImg.length} mit Bild\n`);

  console.log('[2/3] MobileNet v2 laden …');
  const model = await mobilenet.load({ version: 2, alpha: 1.0 });
  console.log('→ Modell bereit\n');

  console.log('[3/3] Embeddings berechnen …');
  const cardIds = [];
  const embeddings = [];
  const t0 = Date.now();
  let skipped = 0;

  for (let i = 0; i < withImg.length; i++) {
    const card = withImg[i];
    try {
      const tensor = await imageToTensor(card.media.image_url);
      const activation = model.infer(tensor, true);
      const arr = await activation.data();
      activation.dispose();
      tensor.dispose();

      // L2-Normalisierung (gleicher Code wie im Browser → matched Cosine-Sim)
      let norm = 0;
      for (let k = 0; k < arr.length; k++) norm += arr[k] * arr[k];
      norm = Math.sqrt(norm);
      const normed = new Array(arr.length);
      for (let k = 0; k < arr.length; k++) {
        // Auf 4 Nachkommastellen runden — JSON-Size halbiert sich, Cosine-Sim
        // bleibt bei < 0.0001 Genauigkeit, völlig ausreichend für Matching
        normed[k] = Math.round((arr[k] / norm) * 10000) / 10000;
      }
      embeddings.push(normed);
      cardIds.push(card.id);

      if ((i + 1) % 20 === 0 || i === withImg.length - 1) {
        const dt = (Date.now() - t0) / 1000;
        const eta = dt / (i + 1) * (withImg.length - i - 1);
        console.log(`  ${i + 1}/${withImg.length}  (${Math.round(dt)}s, ETA ${Math.round(eta)}s)`);
      }
    } catch (err) {
      skipped++;
      console.warn(`  Skip "${card.name}": ${err.message}`);
    }
  }

  const dt = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`→ ${cardIds.length} Embeddings, ${skipped} übersprungen (${dt}s)\n`);

  const output = {
    dim: embeddings.length > 0 ? embeddings[0].length : 0,
    cardIds,
    embeddings
  };
  const json = JSON.stringify(output);
  fs.writeFileSync(OUT_PATH, json);
  const kb = Math.round(fs.statSync(OUT_PATH).size / 1024);
  console.log(`Geschrieben: ${OUT_PATH} (${kb} KB)`);
}

main().catch(err => {
  console.error('FEHLER:', err);
  process.exit(1);
});
