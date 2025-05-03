import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np

from Files.Datasets.dataset import ColorPickerDataset
from additional import hex_to_rgb_tensor


class ColorPickerClusteredDataset(Dataset):
    """
    Dataset z etykietami opartymi na klastrowaniu kolorów wszystkich adnotacji dla danego obrazu.
    Klastry są liczone raz przy inicjalizacji i cache'owane.
    """

    def __init__(self, photos_dir, results_dir, transform=None, n_clusters=3):
        self.base_dataset = ColorPickerDataset(photos_dir, results_dir, transform)
        self.grouped = self.base_dataset.group_by_image()
        self.image_names = list(self.grouped.keys())
        self.photos_dir = photos_dir
        self.transform = transform
        self.n_clusters = n_clusters

        # Precompute clustered targets
        self.target_cache = {}
        for image_name in self.image_names:
            annotations = self.grouped[image_name]
            all_colors = []
            for ann in annotations:
                for hex_code in ann:
                    rgb = hex_to_rgb_tensor(hex_code)
                    all_colors.append(rgb.numpy())

            if not all_colors:
                self.target_cache[image_name] = torch.zeros(3)
                continue

            color_array = np.array(all_colors)
            try:
                kmeans = KMeans(n_clusters=min(self.n_clusters, len(color_array)), n_init=10)
                kmeans.fit(color_array)
                labels, counts = np.unique(kmeans.labels_, return_counts=True)
                dominant_idx = labels[np.argmax(counts)]
                dominant_rgb = kmeans.cluster_centers_[dominant_idx]
                target_color = torch.tensor(dominant_rgb, dtype=torch.float32)
            except Exception as e:
                print(f"[KMeans ERROR] {image_name}: {e}")
                target_color = torch.tensor(color_array[0], dtype=torch.float32)

            self.target_cache[image_name] = target_color

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.photos_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        target_color = self.target_cache[image_name]
        return image, target_color
