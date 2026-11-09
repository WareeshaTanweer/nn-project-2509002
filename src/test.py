import sys
import torch
from torch.utils.data import DataLoader
from dataset import get_datasets
from model import ParkingCNN

model_path = sys.argv[1] if len(sys.argv) > 1 else "models/docker_model.pt"

_, _, test_set, classes = get_datasets()
model = ParkingCNN()
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for x, y in DataLoader(test_set, batch_size=64):
        y_pred += model(x).argmax(1).tolist()
        y_true += y.tolist()

pairs = list(zip(y_true, y_pred))
tp = sum(1 for t, p in pairs if t == 1 and p == 1)
tn = sum(1 for t, p in pairs if t == 0 and p == 0)
fp = sum(1 for t, p in pairs if t == 0 and p == 1)
fn = sum(1 for t, p in pairs if t == 1 and p == 0)

accuracy = (tp + tn) / len(pairs)
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

print("Model:", model_path)
print("Test images:", len(pairs))
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-score :", round(f1, 4))
print("Confusion matrix [rows=true, cols=pred]:")
print([[tn, fp], [fn, tp]])
