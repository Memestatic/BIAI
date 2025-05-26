import os, warnings, logging

# 1) wymusz liczbę rdzeni logicznych
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count())

# 2) filtrowanie warningów z loky
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores"
)

# 3) ucisz logger Loky / joblib
logging.getLogger("loky").setLevel(logging.ERROR)
logging.getLogger("joblib.externals.loky").setLevel(logging.ERROR)


import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from model import SimpleColorPredictor
#from Files.Datasets.tensor_clustered import ColorPickerClusteredDataset
from Files.Datasets.tensor_clustered_lab import ColorPickerClusteredLabMultiDataset

from torchvision import transforms

augmentations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),        # losowe odbicie lustrzane
    transforms.RandomVerticalFlip(p=0.2),          # losowe odbicie pionowe
    transforms.RandomRotation(degrees=15),         # losowy obrót ±15°
    transforms.RandomAffine(
        degrees=0,                                 # bez dodatkowego obrotu
        translate=(0.1, 0.1),                      # przesunięcie do 10% w osi X i Y
        scale=(0.9, 1.1),                          # zmiana skali od 90% do 110%
        shear=10                                   # pochylenie (shear) do 10°
    ),                                            # jitter kolorów
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # perspektywa
    transforms.ToTensor()
])

class WeightedMSELoss(nn.Module):
    def __init__(self, channel_weights):
        super().__init__()
        self.register_buffer('w', torch.tensor(channel_weights).view(1, 3))

    def forward(self, pred, target):
        diff2 = (pred - target)**2
        weighted = diff2 * self.w
        return weighted.mean()
def train_model(photos_dir, results_dir,
                num_colors=1,
                epochs=10, batch_size=8, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Trening na {device} dla num_colors={num_colors}")

    transform = augmentations

    # Dataset filtruje i przycina do (num_colors,3)
    full_ds = ColorPickerClusteredLabMultiDataset(
        photos_dir, results_dir,
        transform=transform,
        num_colors=num_colors,
        n_clusters=num_colors + 2
    )

    # split
    train_n = int(0.8 * len(full_ds))
    val_n   = len(full_ds) - train_n
    train_ds, val_ds = random_split(full_ds, [train_n, val_n])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    # model
    model     = SimpleColorPredictor(num_colors=num_colors).to(device)
    criterion = WeightedMSELoss([1.5, 1.5, 1.5]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        model.train()
        tloss = 0.0
        for imgs, tg in train_loader:
            imgs, tg = imgs.to(device), tg.to(device)
            optimizer.zero_grad()
            pr = model(imgs)
            loss = criterion(pr, tg)
            loss.backward()
            optimizer.step()
            tloss += loss.item()

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for imgs, tg in val_loader:
                imgs, tg = imgs.to(device), tg.to(device)
                pr = model(imgs)
                vloss += criterion(pr, tg).item()

        print(f"Ep {ep}/{epochs}  train={tloss:.4f}  val={vloss:.4f}")

    out = f"saved_model_{num_colors}.pth"
    torch.save(model.state_dict(), out)
    print("Zapisano model:", out)