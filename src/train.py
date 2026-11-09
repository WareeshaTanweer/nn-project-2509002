import argparse, json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import get_loaders
from model import ParkingCNN

p = argparse.ArgumentParser()
p.add_argument("--name", default="baseline")
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--lr", type=float, default=0.001)
p.add_argument("--dropout", type=float, default=0.0)
p.add_argument("--optimizer", default="adam")
p.add_argument("--weight_decay", type=float, default=0.0)
args = p.parse_args()

torch.manual_seed(42)                       # reproducibility
train_loader, val_loader, _, classes = get_loaders()   # test loader ignored here
model = ParkingCNN(dropout=args.dropout)
loss_fn = nn.CrossEntropyLoss()

if args.optimizer == "adam":
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay)

def run_epoch(loader, train):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n

hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val = float("inf")
for epoch in range(1, args.epochs + 1):
    tl, ta = run_epoch(train_loader, True)
    vl, va = run_epoch(val_loader, False)
    for k, v in zip(hist, [tl, vl, ta, va]):
        hist[k].append(v)
    print(f"Epoch {epoch:2d} | train loss {tl:.4f} acc {ta:.3f} | val loss {vl:.4f} acc {va:.3f}")
    if vl < best_val:
        best_val = vl
        torch.save(model.state_dict(), f"models/{args.name}.pt")

json.dump({"args": vars(args), "history": hist}, open(f"models/{args.name}_history.json", "w"))
json.dump({"classes": classes, "img_size": 64, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}, open("models/preprocessing.json", "w"))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="val")
ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout()
plt.savefig(f"models/curves_{args.name}.png")
print("Best val loss:", round(best_val, 4), "| saved models/" + args.name + ".pt")