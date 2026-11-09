from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

DATA = Path("data")
classes = ["empty", "occupied"]

# 1) Count images per class
counts = {c: len(list((DATA / c).glob("*.jpg"))) for c in classes}
print("Images per class:", counts, "| Total:", sum(counts.values()))

# Visualisation 1: class distribution bar chart
plt.figure(figsize=(4, 3))
plt.bar(counts.keys(), counts.values(), color=["green", "red"])
plt.title("Class distribution")
plt.ylabel("Number of images")
plt.tight_layout()
plt.savefig("vis1_class_distribution.png")

# Visualisation 2: sample images from each class
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for row, c in enumerate(classes):
    files = random.sample(list((DATA / c).glob("*.jpg")), 5)
    for col, f in enumerate(files):
        axes[row, col].imshow(Image.open(f))
        axes[row, col].set_title(c)
        axes[row, col].axis("off")
plt.tight_layout()
plt.savefig("vis2_samples.png")

# Visualisation 3: image width and height histogram
widths, heights = [], []
for c in classes:
    for f in random.sample(list((DATA / c).glob("*.jpg")), 300):
        w, h = Image.open(f).size
        widths.append(w)
        heights.append(h)
plt.figure(figsize=(5, 3))
plt.hist(widths, bins=30, alpha=0.6, label="width")
plt.hist(heights, bins=30, alpha=0.6, label="height")
plt.title("Image sizes (300 per class sampled)")
plt.legend()
plt.tight_layout()
plt.savefig("vis3_sizes.png")

print("Saved 3 figures.")