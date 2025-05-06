import os
from Files.Interface.interface import ColorApp
from tkinter import Tk
from trainLoop import train_model    # <--- import funkcji treningowej
from Files.Visualize.visualize import visualize_predictions

# Globalne zmienne ścieżek i transformacje
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..")

PHOTOS_DIR  = os.path.join(PROJECT_ROOT, "Data", "PhotosColorPicker")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Data", "Res_ColorPickerCustomPicker")

def main():
    # 1. Wgląd do danych przed uczeniem.
    # preview_annotations(PHOTOS_DIR, RESULTS_DIR, max_colors=1)

    #2. Uczenie modelu.
    #trenujemy modele dla K=1..5
    # for k in range(1, 6):
    #     print("\n>>> START TRAINING for", k, "colors")
    #     train_model(
    #         photos_dir=PHOTOS_DIR,
    #         results_dir=RESULTS_DIR,
    #         num_colors=k,
    #         epochs=100,
    #         batch_size=8,
    #         lr=0.01
    #     )

        # 3. Wizualizacja wyników.
    # visualize_predictions(num_colors=5, num_samples=10)

    # 4. GUI – wybór własnego obrazu
    root = Tk()
    app = ColorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()