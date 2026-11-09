from pathlib import Path
import hashlib
from PIL import Image

DATA = Path("data")
corrupt, tiny, dups = [], [], []
seen = {}

for f in DATA.rglob("*.jpg"):
    try:
        with Image.open(f) as im:
            im.verify()              # detects broken files
        with Image.open(f) as im:
            w, h = im.size
    except Exception:
        corrupt.append(f)
        continue
    if w < 20 or h < 20:             # very small crops = possible outliers
        tiny.append(f)
    digest = hashlib.md5(f.read_bytes()).hexdigest()
    if digest in seen:               # identical file already seen
        dups.append(f)
    else:
        seen[digest] = f

print("Corrupted images:", len(corrupt))
print("Tiny images (<20 px):", len(tiny))
print("Exact duplicates:", len(dups))