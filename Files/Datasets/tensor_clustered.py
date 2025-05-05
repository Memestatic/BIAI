import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np

from Files.Datasets.dataset import ColorPickerDataset
from Files.additional import hex_to_rgb_tensor

class ColorPickerClusteredDataset(Dataset):
    """
    Zwraca obrazy wraz z tensorami kształtu (num_colors, 3),
    będącymi top-N centrów klastrów posortowanych malejąco po liczebności.
    Obrazy z mniej niż num_colors centrami są odrzucane.
    """
    def __init__(self, photos_dir, results_dir,
                 transform=None,
                 num_colors=3,
                 n_clusters=None):
        # jeśli nie podano, użyjemy n_clusters = num_colors+1
        self.num_colors  = num_colors
        self.n_clusters  = n_clusters or (num_colors + 1)
        self.transform   = transform
        self.photos_dir  = photos_dir

        # bazowy dataset do grupowania adnotacji
        base = ColorPickerDataset(photos_dir, results_dir)
        grouped = base.group_by_image()  # dict: image_name -> list of lists of hex

        # cache i lista zaakceptowanych obrazów
        self.target_cache = {}
        valid = []

        for name, annotations in grouped.items():
            # wektor wszystkich punktów RGB
            all_pts = []
            for ann in annotations:
                for hx in ann:
                    all_pts.append(hex_to_rgb_tensor(hx).numpy())
            if len(all_pts) < self.num_colors:
                continue

            arr = np.stack(all_pts, axis=0)
            k = min(self.n_clusters, len(arr))
            try:
                km = KMeans(n_clusters=k, n_init=10)
                km.fit(arr)
                labels, counts = np.unique(km.labels_, return_counts=True)
                # sortujemy centra malejąco po counts
                idx_sort = np.argsort(-counts)
                centers = km.cluster_centers_[idx_sort]
            except Exception as e:
                # gdyby KMeans padł, bierzemy pierwsze num_colors punktów
                centers = arr[:self.num_colors]

            if centers.shape[0] < self.num_colors:
                continue

            # bierzemy top num_colors i cache'ujemy jako tensor (num_colors,3)
            top = centers[:self.num_colors]
            self.target_cache[name] = torch.tensor(top, dtype=torch.float32)
            valid.append(name)

        self.image_names = valid

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        path = os.path.join(self.photos_dir, name)
        img  = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        tgt = self.target_cache[name]  # (num_colors,3)
        return img, tgt