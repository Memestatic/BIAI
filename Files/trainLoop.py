import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from model import SimpleColorPredictor
from Files.Datasets.tensor_clustered import ColorPickerClusteredDataset

def train_model(photos_dir, results_dir,
                num_colors=1,
                epochs=10, batch_size=8, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Trening na {device} dla num_colors={num_colors}")

    transform = transforms.Compose([
        transforms.Resize((420,420)),
        transforms.ToTensor()
    ])

    # Dataset filtruje i przycina do (num_colors,3)
    full_ds = ColorPickerClusteredDataset(
        photos_dir, results_dir,
        transform=transform,
        num_colors=num_colors,
        n_clusters=num_colors+1
    )

    # split
    train_n = int(0.8 * len(full_ds))
    val_n   = len(full_ds) - train_n
    train_ds, val_ds = random_split(full_ds, [train_n, val_n])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    # model
    model     = SimpleColorPredictor(num_colors=num_colors).to(device)
    criterion = nn.MSELoss()
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