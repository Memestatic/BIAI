import os
import torch
from torch.utils.data import Dataset
from PIL import Image

from Files.additional import hex_to_rgb_tensor
from Files.Datasets.dataset import ColorPickerDataset  # Twoja oryginalna klasa, która zawiera metodę group_by_image()


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

        # Pobieramy pierwszy kolor z pierwszej adnotacji
            annotation_lists = self.grouped[image_name]
            if annotation_lists:
                first_annotation = annotation_lists[0]
                if first_annotation:
                    target_color = hex_to_rgb_tensor(first_annotation[0]) # tensor(3,)
                else:
                    target_color = torch.zeros(3)
            else:
                target_color = torch.zeros(3)

            return image, target_color
