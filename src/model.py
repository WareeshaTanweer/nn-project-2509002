import torch.nn as nn

class ParkingCNN(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   # 3x64x64 -> 16x64x64
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> 16x32x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # -> 32x32x32
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> 32x16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> 64x16x16
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> 64x8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # 64*8*8 = 4096
            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),                            # 2 scores: empty, occupied
        )

    def forward(self, x):
        return self.classifier(self.features(x))

if __name__ == "__main__":
    m = ParkingCNN()
    print(m)
    print("Trainable parameters:", sum(p.numel() for p in m.parameters() if p.requires_grad))