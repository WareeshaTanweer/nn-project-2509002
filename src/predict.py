import sys
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
