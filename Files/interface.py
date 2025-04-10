import os
import torch
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from torchvision import transforms
from model import SimpleColorPredictor

class ColorApp:
    def __init__(self, root=None, model_path="saved_model.pth", img_size=(224, 224)):
        self.model_path = model_path
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])

        self.model = SimpleColorPredictor()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        self.max_display_width = 600  # szerokość ograniczamy
        self.padding_below_image = 100  # miejsce na kwadrat i opis

        if root:
            self.root = root
            self.root.title("Dominujący kolor - Predykcja")
            self.build_gui()

    def build_gui(self):
        self.btn = tk.Button(self.root, text="Wybierz obraz", command=self.load_image)
        self.btn.pack(pady=10)

        # Pusta przestrzeń – będzie modyfikowana dynamicznie
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack()

    def predict_dominant_color(self, image_path):
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)

        rgb = output[0].numpy()
        rgb = [int(c * 255) for c in rgb]
        return rgb

    def resize_to_fit(self, image, max_width):
        img_w, img_h = image.size
        if img_w > max_width:
            ratio = max_width / img_w
            new_size = (int(img_w * ratio), int(img_h * ratio))
            return image.resize(new_size, Image.Resampling.LANCZOS)
        return image

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Obrazy", "*.jpg;*.png;*.jpeg")])
        if not file_path:
            return

        image = Image.open(file_path)
        display_image = self.resize_to_fit(image, self.max_display_width)
        self.tk_image = ImageTk.PhotoImage(display_image)

        img_w = self.tk_image.width()
        img_h = self.tk_image.height()

        # 🔄 Ustawiamy canvas na odpowiedni rozmiar
        total_height = img_h + self.padding_below_image
        self.canvas.config(width=img_w, height=total_height)
        self.canvas.delete("all")

        # Wyśrodkowanie obrazka
        x = (img_w) // 2
        self.canvas.create_image(x, 0, anchor=tk.N, image=self.tk_image)

        # Predykcja koloru
        rgb = self.predict_dominant_color(file_path)
        hex_color = '#%02x%02x%02x' % tuple(rgb)

        # Kwadrat koloru
        square_size = 50
        gap = 20
        square_x = img_w // 2 - square_size // 2
        square_y = img_h + gap

        self.canvas.create_rectangle(
            square_x, square_y,
            square_x + square_size, square_y + square_size,
            fill=hex_color, outline=""
        )

        # Tekst RGB pod kwadratem
        self.canvas.create_text(
            img_w // 2,
            square_y + square_size + 20,
            text=f"Dominujący kolor: RGB{tuple(rgb)}",
            font=("Arial", 12)
        )

    def run(self):
        self.root.mainloop()
