"""
Handy-realistische Augmentationen fuer Riftbound-Karten-Training.

Simuliert die Domain-Luecke zwischen Studio-Bild (flach, perfekt beleuchtet)
und echtem Handy-Foto (Winkel, Glanz, Rauschen, JPEG-Artefakte, Beleuchtung).

Alle Transforms: PIL.Image rein -> PIL.Image raus, damit sie in eine
torchvision-Compose-Pipeline VOR ToImage() passen.
"""
import random
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from io import BytesIO


class JPEGCompression:
    """Echte Handy-Fotos sind JPEG-komprimiert -> Block-Artefakte, Farbmatsch.
    Studio-PNGs haben das nie. Wichtigster Domain-Gap-Faktor."""
    def __init__(self, p=0.6, quality=(28, 88)):
        self.p, self.quality = p, quality
    def __call__(self, img):
        if random.random() > self.p:
            return img
        q = random.randint(*self.quality)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class SensorNoise:
    """Kamera-Sensor-Rauschen, staerker bei wenig Licht."""
    def __init__(self, p=0.5, sigma=(2, 14)):
        self.p, self.sigma = p, sigma
    def __call__(self, img):
        if random.random() > self.p:
            return img
        arr = np.asarray(img).astype(np.float32)
        s = random.uniform(*self.sigma)
        noise = np.random.normal(0, s, arr.shape).astype(np.float32)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


class WhiteBalanceShift:
    """Handy-Weissabgleich schwankt (warmes Zimmerlicht vs kaltes LED)."""
    def __init__(self, p=0.5, strength=0.12):
        self.p, self.strength = p, strength
    def __call__(self, img):
        if random.random() > self.p:
            return img
        arr = np.asarray(img).astype(np.float32)
        # Zufaellige Kanal-Multiplikatoren (Rot warm, Blau kalt)
        rmul = 1.0 + random.uniform(-self.strength, self.strength)
        bmul = 1.0 + random.uniform(-self.strength, self.strength)
        arr[..., 0] *= rmul
        arr[..., 2] *= bmul
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class LightingGradient:
    """Ungleichmaessige Beleuchtung: eine Kartenseite heller als die andere.
    Sehr typisch fuer Handheld-Scans (Lichtquelle seitlich)."""
    def __init__(self, p=0.55, strength=(0.15, 0.45)):
        self.p, self.strength = p, strength
    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        arr = np.asarray(img).astype(np.float32)
        angle = random.uniform(0, 2 * np.pi)
        strength = random.uniform(*self.strength)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # Normierter Gradient entlang zufaelliger Richtung
        grad = (np.cos(angle) * (xx / w - 0.5) + np.sin(angle) * (yy / h - 0.5))
        grad = grad / (np.abs(grad).max() + 1e-6)  # -1..1
        mult = (1.0 + strength * grad)[..., None]
        return Image.fromarray(np.clip(arr * mult, 0, 255).astype(np.uint8))


class DirectionalFoilGlare:
    """Realistischer Foil-Glanz: gerichtetes helles Band mit leichtem
    Farbstich (Regenbogen-Highlight), nicht nur runde weisse Flecken."""
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        arr = np.asarray(img).astype(np.float32)
        angle = random.uniform(0, np.pi)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # Abstand zu einer zufaelligen Linie -> schmales helles Band
        cx, cy = w * random.uniform(0.2, 0.8), h * random.uniform(0.2, 0.8)
        d = np.abs(np.cos(angle) * (xx - cx) + np.sin(angle) * (yy - cy))
        band_w = random.uniform(h * 0.05, h * 0.22)
        glow = np.exp(-(d ** 2) / (2 * band_w ** 2))
        intensity = random.uniform(60, 160)
        # Leichter Farbstich pro Kanal (Foil-Schimmer)
        tint = np.array([random.uniform(0.85, 1.0),
                         random.uniform(0.85, 1.0),
                         random.uniform(0.85, 1.0)], dtype=np.float32)
        add = (glow[..., None] * intensity) * tint
        return Image.fromarray(np.clip(arr + add, 0, 255).astype(np.uint8))


class MotionBlur:
    """Verwackeltes Handheld-Foto: gerichtete Bewegungsunschaerfe."""
    def __init__(self, p=0.25, max_len=9):
        self.p, self.max_len = p, max_len
    def __call__(self, img):
        if random.random() > self.p:
            return img
        length = random.randint(3, self.max_len)
        angle = random.uniform(0, 360)
        # Linien-Kernel bauen
        k = np.zeros((length, length), dtype=np.float32)
        k[length // 2, :] = 1.0
        # rotieren via PIL
        kimg = Image.fromarray((k * 255).astype(np.uint8)).rotate(angle)
        kern = np.asarray(kimg).astype(np.float32)
        kern /= (kern.sum() + 1e-6)
        arr = np.asarray(img).astype(np.float32)
        # separable-ish: nutze PIL Kernel-Filter fuer 3x3..5x5, sonst gaussian-Fallback
        if length <= 5:
            flat = (kern / kern.sum()).flatten().tolist()
            size = kern.shape[0]
            try:
                return img.filter(ImageFilter.Kernel((size, size), flat, scale=1))
            except Exception:
                pass
        return img.filter(ImageFilter.GaussianBlur(radius=length / 4.0))


class ChromaticAberration:
    """Guenstige Handy-Optik: Farbraender an Kanten (R/B leicht verschoben)."""
    def __init__(self, p=0.3, max_shift=3):
        self.p, self.max_shift = p, max_shift
    def __call__(self, img):
        if random.random() > self.p:
            return img
        arr = np.asarray(img)
        sx = random.randint(-self.max_shift, self.max_shift)
        sy = random.randint(-self.max_shift, self.max_shift)
        r = np.roll(arr[..., 0], (sy, sx), axis=(0, 1))
        b = np.roll(arr[..., 2], (-sy, -sx), axis=(0, 1))
        out = arr.copy()
        out[..., 0] = r
        out[..., 2] = b
        return Image.fromarray(out)


def build_phone_pipeline(image_size, torchvision_T, torch):
    """Baut die komplette Train-Pipeline mit Handy-Realismus.
    torchvision_T = torchvision.transforms.v2, torch = torch modul."""
    T = torchvision_T
    return T.Compose([
        T.Resize((int(image_size * 1.20), int(image_size * 1.20))),
        T.RandomCrop(image_size),
        T.RandomRotation(degrees=18, fill=128),
        T.RandomPerspective(distortion_scale=0.28, p=0.55, fill=128),
        # --- Handy-spezifisch (PIL-basiert) ---
        LightingGradient(p=0.55),
        DirectionalFoilGlare(p=0.5),
        WhiteBalanceShift(p=0.5),
        MotionBlur(p=0.25),
        ChromaticAberration(p=0.3),
        SensorNoise(p=0.5),
        JPEGCompression(p=0.6),
        # --- torchvision Standard ---
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.05),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.RandomErasing(p=0.2, scale=(0.02, 0.12)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_phone_val_pipeline(image_size, torchvision_T, torch):
    """Moderate, DETERMINISTISCHE Handy-Augmentation fuer Validierung.
    Misst handy-aehnliche Genauigkeit (im Gegensatz zur clean-val)."""
    T = torchvision_T
    return T.Compose([
        T.Resize((image_size, image_size)),
        LightingGradient(p=1.0, strength=(0.2, 0.25)),
        JPEGCompression(p=1.0, quality=(45, 55)),
        SensorNoise(p=1.0, sigma=(6, 6)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
