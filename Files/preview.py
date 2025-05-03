import os
import matplotlib.pyplot as plt
from PIL import Image
from Files.Datasets.dataset import ColorPickerDataset

def preview_annotations(photos_dir, results_dir, max_colors=1):
    """
    Wyświetla obrazy i ich adnotacje, ograniczając do tych, które mają <= max_colors.
    """
    dataset = ColorPickerDataset(photos_dir, results_dir)
    grouped = dataset.group_by_image()

    for obraz_count, (image_name, annotations) in enumerate(grouped.items(), start=1):
        image_path = os.path.join(photos_dir, image_name)
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Obraz {image_name} nie został znaleziony!")
            continue

        sample_count = 0
        for colors in annotations:
            if 0 < len(colors) <= max_colors:
                sample_count += 1

                fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [8, 1]}, figsize=(6, 8))
                ax1.imshow(image)
                ax1.axis('off')

                ax2.axis('off')
                ax2.text(0.5, 0.5,
                         f"Obraz {obraz_count}, Próbka {sample_count}\nKolory: {colors}",
                         ha='center', va='center', fontsize=12, wrap=True)

                plt.tight_layout()
                plt.show()
