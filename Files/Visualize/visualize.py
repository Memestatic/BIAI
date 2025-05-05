import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from Files.model import SimpleColorPredictor
from Files.Datasets.tensor_clustered import ColorPickerClusteredDataset

# katalog główny projektu
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)
PHOTOS_DIR  = os.path.join(BASE_DIR, "Data", "PhotosColorPicker")
RESULTS_DIR = os.path.join(BASE_DIR, "Data", "Res_ColorPickerCustomPicker")
MODELS_DIR  = os.path.join(BASE_DIR, "Files")  # tu są saved_model_{k}.pth

def show_color_patch(ax, color_tensor, title):
    rgb = color_tensor.detach().cpu().numpy()
    rgb = [max(0, min(1, c)) for c in rgb]
    ax.imshow([[rgb]])
    ax.set_title(title)
    ax.axis('off')

def visualize_predictions(
    num_colors: int = 1,
    num_samples: int = 5
):
    """
    Dla każdej z num_samples próbek rysuje:
     – obrazek (pełna szerokość),
     – przewidywane kolory (rząd 1),
     – prawdziwe kolory    (rząd 2).
    """
    # 1) przygotuj zbiór walidacyjny
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    full_ds = ColorPickerClusteredDataset(
        PHOTOS_DIR, RESULTS_DIR,
        transform=transform,
        num_colors=num_colors,
        n_clusters=num_colors+1
    )
    train_n = int(0.8 * len(full_ds))
    _, val_ds = random_split(full_ds, [train_n, len(full_ds) - train_n])
    loader = DataLoader(val_ds, batch_size=1, shuffle=True)

    # 2) załaduj model
    model = SimpleColorPredictor(num_colors=num_colors)
    model_path = os.path.join(MODELS_DIR, f"saved_model_{num_colors}.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"[INFO] Załadowano model ze ścieżki: {model_path}")
    else:
        print(f"[WARNING] Nie znaleziono pliku modelu pod ścieżką: {model_path}")
    model.eval()

    # 3) rysuj
    shown = 0
    with torch.no_grad():
        for img, tgt in loader:
            out = model(img)

            # odwrócenie Tensora do obrazu
            img_np = img[0].permute(1, 2, 0).cpu().numpy()

            # GridSpec: 3 rzędy (obraz, pred, act), num_colors kolumn
            fig = plt.figure(figsize=(2*num_colors, 6))
            gs = fig.add_gridspec(3, num_colors, height_ratios=[4,1,1])

            # rząd 0: obraz na całą szerokość
            ax_img = fig.add_subplot(gs[0, :])
            ax_img.imshow(img_np)
            ax_img.axis('off')
            ax_img.set_title(f"Próbka #{shown+1}")

            # rzędy 1 i 2: kolory
            for i in range(num_colors):
                ax_pred = fig.add_subplot(gs[1, i])
                show_color_patch(ax_pred, out[0, i], f"P{i+1}-pred")

                ax_act = fig.add_subplot(gs[2, i])
                show_color_patch(ax_act, tgt[0, i], f"P{i+1}-act")

            plt.tight_layout()
            plt.show()

            shown += 1
            if shown >= num_samples:
                break