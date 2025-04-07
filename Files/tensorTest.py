import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms

from dataset import ColorPickerDataset  # upewnij się, że ścieżka importu jest prawidłowa
import additional
from tensor import ColorPickerTensorDataset  # zaimportuj swoją klasę


def test_tensor_dataset():
    # Ustalamy ścieżki
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..")
    photos_dir = os.path.join(project_root, "Data", "PhotosColorPicker")
    results_dir = os.path.join(project_root, "Data", "Res_ColorPickerCustomPicker")

    # Definiujemy transformacje dla obrazu (np. zmiana rozmiaru i konwersja do tensora)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Tworzymy instancję datasetu
    tensor_dataset = ColorPickerTensorDataset(photos_dir, results_dir, transform=transform)

    # Pobieramy jedną próbkę np. pod indeksem 0
    sample = tensor_dataset[0]  # sample to słownik z kluczami: "image", "annotations", "image_name"

    # Sprawdzamy zwracane elementy
    image_tensor = sample["image"]
    annotations = sample["annotations"]
    image_name = sample["image_name"]

    print("Nazwa obrazu:", image_name)
    print("Typ obrazu:", type(image_tensor))
    print("Kształt obrazu:", image_tensor.shape)

    # Anotacje to lista tensorów – wypiszemy je
    for i, ann in enumerate(annotations, start=1):
        print(f"Adnotacja {i}:")
        print("  Typ:", type(ann))
        print("  Kształt:", ann.shape)
        # Opcjonalnie: wypisanie wartości
        print("  Wartości:", ann)


if __name__ == "__main__":
    test_tensor_dataset()
