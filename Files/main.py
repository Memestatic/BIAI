import os

from Files.visualize import visualize_predictions
from trainLoop import train_model
from preview import preview_annotations
from interface import ColorApp  # nowa funkcja do odpalenia GUI
from tkinter import Tk

# 🔧 Globalne zmienne ścieżek i transformacje
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..")

PHOTOS_DIR = os.path.join(PROJECT_ROOT, "Data", "PhotosColorPicker")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Data", "Res_ColorPickerCustomPicker")

def main():
    # Wybierz jedną z opcji:

    # 1. Wgląd do danych przed uczeniem.
     #preview_annotations(PHOTOS_DIR, RESULTS_DIR, max_colors=1)

    # 2. Uczenie modelu.
     #train_model(PHOTOS_DIR, RESULTS_DIR, epochs=300, lr=0.0005, batch_size=8)

    # 3. Wizualizacja wyników.
     #visualize_predictions(PHOTOS_DIR, RESULTS_DIR, model_path="saved_model.pth", num_samples=10)

    # 4. GUI – wybór własnego obrazu
    root = Tk()
    app = ColorApp(root=root, model_path="saved_model.pth", img_size=(224, 224))
    app.run()

if __name__ == "__main__":
    main()
