import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from dataset import ColorPickerDataset  # Twoja oryginalna klasa, która zawiera metodę group_by_image()
from additional import process_annotation


class ColorPickerTensorDataset(Dataset):
    """
    Nowa klasa, która dla każdego obrazu zwraca:
      - obraz jako tensor (np. z kształtem (3, H, W)),
      - adnotacje w formie listy tensorów. Każdy tensor odpowiada wyborom jednego studenta
        i ma kształt (N, 3), gdzie N zależy od liczby wybranych kolorów.
      - nazwę obrazu (opcjonalnie)
    """

    def __init__(self, photos_dir, results_dir, transform=None):
        # Korzystamy z istniejącej klasy, która wczytuje dane z plików
        self.base_dataset = ColorPickerDataset(photos_dir, results_dir, transform)
        # Grupujemy próbki według obrazu – metodą group_by_image() z oryginalnej klasy
        self.grouped = self.base_dataset.group_by_image()
        # Tworzymy listę kluczy (nazw obrazów), aby umożliwić indeksowanie
        self.image_names = list(self.grouped.keys())
        self.photos_dir = photos_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Pobieramy nazwę obrazu według indeksu
        image_name = self.image_names[idx]
        image_path = os.path.join(self.photos_dir, image_name)
        # Wczytujemy obraz z dysku
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Pobieramy wszystkie adnotacje dla tego obrazu – lista list kodów hex
        annotations_lists = self.grouped[image_name]
        # Dla każdej adnotacji przetwarzamy ją funkcją process_annotation,
        # dzięki czemu uzyskujemy tensor o wymiarach (N, 3)
        annotations_tensors = [process_annotation(annotation) for annotation in annotations_lists]

        return {
            "image": image,  # Tensor obrazu o wymiarach (3, H, W)
            "annotations": annotations_tensors,  # Lista tensorów (każdy: (N, 3))
            "image_name": image_name
        }
