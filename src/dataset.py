import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

IMG_SIZE = 64
SEED = 42

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),            # same size for every image
    transforms.ToTensor(),                              # pixels 0-255 -> 0-1
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 0-1 -> -1 to 1
])

def get_datasets(data_dir="data"):
    full = datasets.ImageFolder(data_dir, transform=transform)
    n = len(full)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    g = torch.Generator().manual_seed(SEED)            # fixed seed = same split every time
    train, val, test = random_split(full, [n_train, n_val, n_test], generator=g)
    return train, val, test, full.classes

def get_loaders(data_dir="data", batch_size=64):
    train, val, test, classes = get_datasets(data_dir)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
        classes,
    )

if __name__ == "__main__":
    train, val, test, classes = get_datasets()
    print("Classes:", classes)
    print("Train / Val / Test:", len(train), len(val), len(test))
    x, y = train[0]
    print("One image tensor shape:", x.shape, "label:", y)