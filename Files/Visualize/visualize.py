import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from model import SimpleColorPredictor
from Files.Datasets.tensor_clustered import ColorPickerClusteredDataset as ColorPickerTensorDataset

def show_color_patch(ax, color_tensor, title):
    """ Pomocnicza funkcja do rysowania prostokąta koloru RGB """
    rgb = color_tensor.detach().cpu().numpy()
    rgb = [max(0, min(1, c)) for c in rgb]  # upewniamy się, że są w zakresie [0,1]
    ax.imshow([[rgb]])
    ax.set_title(title)
    ax.axis('off')


def visualize_predictions(photos_dir, results_dir, model_path=None, num_samples=5):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    full_dataset = ColorPickerTensorDataset(photos_dir, results_dir, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    model = SimpleColorPredictor()
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Model załadowany z:", model_path)
    model.eval()

    shown = 0
    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images)
            predicted = outputs[0]
            actual = targets[0]

            fig, axes = plt.subplots(1, 3, figsize=(10, 3))

            # Oryginalny obraz
            axes[0].imshow(images[0].permute(1, 2, 0))
            axes[0].set_title("Obraz")
            axes[0].axis('off')

            # Przewidziany kolor
            show_color_patch(axes[1], predicted, "Przewidziany")

            # Rzeczywisty kolor
            show_color_patch(axes[2], actual, "Rzeczywisty")

            plt.tight_layout()
            plt.show()

            shown += 1
            if shown >= num_samples:
                break
