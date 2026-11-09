import random
import shutil
from pathlib import Path

# CHANGE THIS if your folder is in a different place
SOURCE = Path(r"C:\datasets\archive\PKLot\PKLotSegmented")
OUT = Path("data")
PER_CLASS = 2500          # images per class
random.seed(42)           # fixed seed so results are reproducible

for cls in ["Empty", "Occupied"]:
    # find every .jpg whose parent folder is named Empty / Occupied
    files = [p for p in SOURCE.rglob("*.jpg") if p.parent.name.lower() == cls.lower()]
    print(cls, "found:", len(files))
    chosen = random.sample(files, PER_CLASS)

    out_dir = OUT / cls.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(chosen):
        shutil.copy(f, out_dir / f"{cls.lower()}_{i}.jpg")

print("Done. Subset saved in", OUT)