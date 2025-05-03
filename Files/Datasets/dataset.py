import os
from collections import defaultdict

from PIL import Image
import torch
from torch.utils.data import Dataset


class ColorPickerDataset(Dataset):
    def __init__(self, photos_dir, results_dir, transform=None):
        self.photos_dir = photos_dir  # Ścieżka do folderu z obrazami
        self.results_dir = results_dir  # Ścieżka do folderu z plikami tekstowymi
        self.transform = transform

        # Lista przechowująca wszystkie próbki: każda próbka to słownik z kluczami np. 'image' i 'colors'
        self.samples = []

        # Iteracja przez wszystkie pliki tekstowe (każdy odpowiada jednemu użytkownikowi)
        for result_file in os.listdir(self.results_dir):
            if result_file.endswith('.txt'):
                if "Time" in result_file:
                    continue
                file_path = os.path.join(self.results_dir, result_file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # Zakładamy, że linijka zaczyna się od nazwy pliku obrazu, a potem następują kolory
                        parts = line.split()
                        image_name = parts[0]

                        # Reszta linijki zawiera kolory – zakładamy, że są oddzielone spacjami lub przecinkami
                        # Jeśli kolory są zapisane jako "#af885a, #af7a3e", możemy rozdzielić je dodatkowo
                        raw_colors = " ".join(parts[1:]).replace(',', ' ')
                        colors = [col.strip() for col in raw_colors.split() if col.strip()]

                        self.samples.append({
                            'image': image_name,
                            'colors': colors
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = os.path.join(self.photos_dir, sample['image'])

        # Wczytanie obrazu
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        colors = sample['colors']

        return image, colors

    def group_by_image(self):
        """
        Grupuje próbki według nazwy obrazu.
        Zwraca słownik, w którym kluczem jest nazwa obrazu, a wartością lista adnotacji kolorów.
        """
        grouped = defaultdict(list)
        for sample in self.samples:
            image_name = sample['image']
            grouped[image_name].append(sample['colors'])
        return dict(grouped)