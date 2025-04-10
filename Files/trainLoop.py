import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from model import SimpleColorPredictor
from tensor_clustered import ColorPickerClusteredDataset as DatasetClass

def train_model(photos_dir, results_dir, epochs=10, batch_size=8, lr=1e-3):
    transform = transforms.Compose([
        transforms.Resize((420, 420)),
        transforms.ToTensor()
    ])

    full_dataset = DatasetClass(photos_dir, results_dir, transform=transform)

    # 🧪 Podział 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = SimpleColorPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 🔍 Walidacja po epoce
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        print(f"Epoka {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "saved_model.pth")
    print("Model zapisany do: saved_model.pth")
