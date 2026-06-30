// Berechnet ORB-Keypoints + Deskriptoren für alle Riftbound-Karten.
// Output: riftbound-orb.json mit Base64-encoded Binärdaten der Deskriptoren.
//
// Format:
// {
//   "version": 1,
//   "descSize": 32,          // Bytes pro Deskriptor (256 bit ORB)
//   "maxKeypoints": 250,
//   "cardIds": ["abc...", ...],
//   "keypointCounts": [200, 250, 85, ...],   // pro Karte tatsächlich extrahiert
//   "descriptorsBase64": "AbCd..."           // alle Deskriptoren konkateniert
// }

const jsfeat = require('jsfeat');
const { Jimp } = require('jimp');
const fs = require('fs');
const path = require('path');

const API_BASE = 'https://api.riftcodex.com';
const OUT_PATH = path.join(__dirname, '..', 'riftbound-orb.json');

// Wir extrahieren max. 250 Keypoints pro Karte. Genug für robustes Matching,
// hält Match-Zeit pro Scan unter 2 Sekunden im Browser.
const MAX_KEYPOINTS = 250;
const FAST_THRESHOLD = 20;
const FAST_BORDER = 5;

// jsfeat braucht ein vorallokiertes Corner-Pool. Manche detail-reichen Karten
// triggern >5000 FAST-Corners → 20000 als sichere Reserve.
const CORNER_POOL_SIZE = 20000;
const cornerPool = [];
for (let i = 0; i < CORNER_POOL_SIZE; i++) cornerPool.push(new jsfeat.keypoint_t());

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

async function loadGrayscaleMatrix(url) {
  const img = await Jimp.read(url);
  const W = img.bitmap.width, H = img.bitmap.height;
  const m = new jsfeat.matrix_t(W, H, jsfeat.U8_t | jsfeat.C1_t);
  const buf = img.bitmap.data;
  for (let i = 0, j = 0; i < buf.length; i += 4, j++) {
    m.data[j] = (0.299 * buf[i] + 0.587 * buf[i+1] + 0.114 * buf[i+2]) | 0;
  }
  return m;
}

function extractORB(matrix, maxKp) {
  jsfeat.fast_corners.set_threshold(FAST_THRESHOLD);
  const numCorners = jsfeat.fast_corners.detect(matrix, cornerPool, FAST_BORDER);
  let useN = Math.min(numCorners, maxKp);

  // Falls mehr Corner als maxKp → nach FAST-Score sortieren (höchste zuerst)
  if (numCorners > maxKp) {
    // Sortier-Block: nur die ersten numCorners interessieren
    const subset = cornerPool.slice(0, numCorners);
    subset.sort((a, b) => b.score - a.score);
    // Top maxKp zurück in Pool kopieren (in-place auf Werten)
    for (let i = 0; i < maxKp; i++) {
      cornerPool[i].x = subset[i].x;
      cornerPool[i].y = subset[i].y;
      cornerPool[i].score = subset[i].score;
      cornerPool[i].angle = subset[i].angle;
      cornerPool[i].level = subset[i].level;
    }
    useN = maxKp;
  }

  const descriptors = new jsfeat.matrix_t(32, useN, jsfeat.U8_t | jsfeat.C1_t);
  jsfeat.orb.describe(matrix, cornerPool, useN, descriptors);
  return { count: useN, descBuffer: Buffer.from(descriptors.data.buffer, 0, useN * 32) };
}

async function main() {
  console.log('[1/3] Karten-Liste laden …');
  const cards = await fetchAllCards();
  const withImg = cards.filter(c => c.media && c.media.image_url && c.id);
  console.log(`→ ${cards.length} Karten, ${withImg.length} mit Bild\n`);

  console.log('[2/3] ORB-Features extrahieren …');
  const cardIds = [];
  const keypointCounts = [];
  const descBuffers = [];
  const t0 = Date.now();
  let skipped = 0;

  for (let i = 0; i < withImg.length; i++) {
    const card = withImg[i];
    try {
      const mat = await loadGrayscaleMatrix(card.media.image_url);
      const orb = extractORB(mat, MAX_KEYPOINTS);
      cardIds.push(card.id);
      keypointCounts.push(orb.count);
      descBuffers.push(orb.descBuffer);

      if ((i + 1) % 25 === 0 || i === withImg.length - 1) {
        const dt = (Date.now() - t0) / 1000;
        const eta = Math.round(dt / (i + 1) * (withImg.length - i - 1));
        console.log(`  ${i + 1}/${withImg.length}  (${Math.round(dt)}s, ETA ${eta}s) — last: ${orb.count} kp`);
      }
    } catch (e) {
      skipped++;
      console.warn(`  Skip "${card.name}": ${e.message}`);
    }
  }

  const totalKp = keypointCounts.reduce((a, b) => a + b, 0);
  const avgKp = (totalKp / keypointCounts.length).toFixed(0);
  console.log(`→ ${cardIds.length} Karten verarbeitet, ${skipped} übersprungen`);
  console.log(`  Gesamt ${totalKp} Keypoints, Schnitt ${avgKp} pro Karte`);

  console.log('\n[3/3] Output schreiben …');
  const fullDescBuffer = Buffer.concat(descBuffers);
  const output = {
    version: 1,
    descSize: 32,
    maxKeypoints: MAX_KEYPOINTS,
    cardIds,
    keypointCounts,
    descriptorsBase64: fullDescBuffer.toString('base64')
  };
  const json = JSON.stringify(output);
  fs.writeFileSync(OUT_PATH, json);
  const mb = (fs.statSync(OUT_PATH).size / 1024 / 1024).toFixed(2);
  console.log(`→ ${OUT_PATH} (${mb} MB)`);
}

main().catch(err => { console.error('FEHLER:', err); process.exit(1); });
