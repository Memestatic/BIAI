import os
import matplotlib.pyplot as plt
from PIL import Image
from dataset import ColorPickerDataset  # upewnij się, że ścieżka importu jest prawidłowa
import additional

def main():
    # Folder, w którym znajduje się main.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Przejście "w górę" o jeden katalog, żeby wyjść z folderu Files do folderu BIAI
    project_root = os.path.join(current_dir, "..")

    photos_dir = os.path.join(project_root, "Data", "PhotosColorPicker")
    results_dir = os.path.join(project_root, "Data", "Res_ColorPickerCustomPicker")

    # Tworzymy instancję datasetu
    dataset = ColorPickerDataset(photos_dir, results_dir)
    print("Liczba próbek w dataset:", len(dataset))

    # Grupujemy próbki według nazwy obrazu
    grouped = dataset.group_by_image()  # metoda zwraca słownik: { image_name: [lista adnotacji] }

    obraz_count = 0
    # Iterujemy po pogrupowanych danych
    for image_name, annotations in grouped.items():
        obraz_count += 1

        # Budujemy pełną ścieżkę do obrazu
        image_path = os.path.join(photos_dir, image_name)
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Obraz {image_name} nie został znaleziony!")
            continue

        sample_count = 0
        # Iterujemy po wszystkich adnotacjach dla danego obrazu
        for colors in annotations:
            sample_count += 1

            # Tworzymy figurę z dwoma subplots: górny na obraz, dolny na tekst
            fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [8, 1]}, figsize=(6, 8))
            ax1.imshow(image)
            ax1.axis('off')

            ax2.axis('off')
            ax2.text(0.5, 0.5,
                     f"Obraz {obraz_count}, Próbka {sample_count}\nKolory: {colors}",
                     ha='center', va='center', fontsize=12, wrap=True)

            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()
