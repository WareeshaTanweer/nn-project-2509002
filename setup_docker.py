from pathlib import Path

files = {}

files["requirements-train.txt"] = """torch==2.15.0
torchvision==0.29.2
numpy==2.5.4
pillow==13.0.0
matplotlib==3.11.2
"""

files["requirements-test.txt"] = """torch==2.15.0
torchvision==0.29.2
numpy==2.5.4
pillow==13.0.0
"""

files["Dockerfile.train"] = """FROM python:3.13-slim
WORKDIR /app
COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY src/ ./src/
ENV PYTHONUNBUFFERED=1
CMD ["python", "src/train.py", "--name", "docker_model", "--dropout", "0.5"]
"""

files["Dockerfile.test"] = """FROM python:3.13-slim
WORKDIR /app
COPY requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-test.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY src/test.py src/predict.py src/model.py src/dataset.py ./src/
ENV PYTHONUNBUFFERED=1
CMD ["python", "src/test.py"]
"""

files[".dockerignore"] = """data
models
*.png
__pycache__
"""

files["src/test.py"] = """import sys
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
"""

files["src/predict.py"] = """import sys
import json
import torch
from PIL import Image
from dataset import transform
from model import ParkingCNN

img_path = sys.argv[1]
model_path = sys.argv[2] if len(sys.argv) > 2 else "models/docker_model.pt"

try:
    classes = json.load(open("models/preprocessing.json"))["classes"]
except FileNotFoundError:
    classes = ["empty", "occupied"]

model = ParkingCNN()
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

img = Image.open(img_path).convert("RGB")
x = transform(img).unsqueeze(0)
with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0]
idx = int(probs.argmax())
print(f"Prediction: {classes[idx]} (confidence {probs[idx]:.1%})")
"""

for name, text in files.items():
    Path(name).write_text(text, encoding="utf-8")
    print("Created", name)

# patch src/train.py (two small edits)
p = Path("src/train.py")
t = p.read_text(encoding="utf-8")
t = t.replace('plt.savefig(f"curves_{args.name}.png")',
              'plt.savefig(f"models/curves_{args.name}.png")')
marker = 'json.dump({"args": vars(args), "history": hist}, open(f"models/{args.name}_history.json", "w"))'
extra = 'json.dump({"classes": classes, "img_size": 64, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}, open("models/preprocessing.json", "w"))'
if "preprocessing.json" not in t and marker in t:
    t = t.replace(marker, marker + "\n" + extra)
p.write_text(t, encoding="utf-8")
print("Patched src/train.py")