import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
from skimage import color

from Files.Datasets.dataset import ColorPickerDataset
from Files.additional import hex_to_rgb_tensor


class ColorPickerClusteredLabMultiDataset(Dataset):
    """
    Dataset zwracający (obraz, tensor [num_colors, 3]) – top-N kolorów w przestrzeni Lab.
    """
    def __init__(self, photos_dir, results_dir, transform=None, num_colors=3, n_clusters=None):
        self.num_colors = num_colors
        self.n_clusters = n_clusters or (num_colors + 1)
        self.transform = transform
        self.photos_dir = photos_dir

        base = ColorPickerDataset(photos_dir, results_dir)
        grouped = base.group_by_image()

        self.target_cache = {}
        valid_images = []

        for name, annotations in grouped.items():
            all_pts = []
            for ann in annotations:
                for hx in ann:
                    rgb = hex_to_rgb_tensor(hx)
                    lab = self.rgb_to_lab_tensor(rgb)
                    all_pts.append(lab.numpy())

            if len(all_pts) < self.num_colors:
                continue

            arr = np.stack(all_pts, axis=0)
            k = min(self.n_clusters, len(arr))
            try:
                km = KMeans(n_clusters=k, n_init=10)
                km.fit(arr)
                labels, counts = np.unique(km.labels_, return_counts=True)
                idx_sort = np.argsort(-counts)
                centers = km.cluster_centers_[idx_sort]
            except Exception as e:
                centers = arr[:self.num_colors]

            if centers.shape[0] < self.num_colors:
                continue

            top = centers[:self.num_colors]
            self.target_cache[name] = torch.tensor(top, dtype=torch.float32)
            valid_images.append(name)

        self.image_names = valid_images

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        path = os.path.join(self.photos_dir, name)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        tgt = self.target_cache[name]  # shape: [num_colors, 3]
        return img, tgt

    @staticmethod
    def rgb_to_lab_tensor(rgb_tensor):
        rgb = rgb_tensor.numpy().reshape(1, 1, 3)
        lab = color.rgb2lab(rgb)
        l, a, b = lab.flatten()
        return torch.tensor([l / 100, (a + 128) / 255, (b + 128) / 255], dtype=torch.float32)
