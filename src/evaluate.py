import sys
import math
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)
from dataset import get_datasets
from model import ParkingCNN

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "models/exp1_dropout.pt"

_, val_set, test_set, classes = get_datasets()
model = ParkingCNN(dropout=0.5)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

def predict(subset):
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in DataLoader(subset, batch_size=64):
            y_pred += model(x).argmax(1).tolist()
            y_true += y.tolist()
    paths = [subset.dataset.samples[i][0] for i in subset.indices]
    return y_true, y_pred, paths

y_true, y_pred, paths = predict(test_set)

# ---- metrics (positive class = occupied = 1) ----
print("Test images:", len(y_true))
print("Accuracy :", round(accuracy_score(y_true, y_pred), 4))
print("Precision:", round(precision_score(y_true, y_pred), 4))
print("Recall   :", round(recall_score(y_true, y_pred), 4))
print("F1-score :", round(f1_score(y_true, y_pred), 4))
majority = max(y_true.count(0), y_true.count(1)) / len(y_true)
print("Baseline (always predict most common class):", round(majority, 4))
print("Confusion matrix [rows=true, cols=pred]:")
print(confusion_matrix(y_true, y_pred))

ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=classes)
plt.title("Confusion matrix (test set)")
plt.savefig("confusion_matrix.png")
plt.close()

# ---- wrong predictions ----
wrong = [(p, t, q, "test") for p, t, q in zip(paths, y_true, y_pred) if t != q]
print("Wrong predictions on test set:", len(wrong))

if len(wrong) < 5:   # assignment wants at least 5 examples
    vt, vp, vpaths = predict(val_set)
    extra = [(p, t, q, "val") for p, t, q in zip(vpaths, vt, vp) if t != q]
    wrong += extra[: 8 - len(wrong)]
    print("Fewer than 5 test errors, added validation-set errors (labelled 'val').")

n = min(len(wrong), 8)
cols = 4
rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows), squeeze=False)
for ax in axes.flat:
    ax.axis("off")
for ax, (path, t, q, src) in zip(axes.flat, wrong[:n]):
    ax.imshow(Image.open(path))
    ax.set_title(f"{src}: true={classes[t]}\npred={classes[q]}", fontsize=9)
plt.tight_layout()
plt.savefig("wrong_predictions.png")
print("Saved confusion_matrix.png and wrong_predictions.png")