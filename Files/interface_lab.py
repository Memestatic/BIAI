import os
import torch
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from torchvision import transforms
from model import SimpleColorPredictor
from skimage import color
import numpy as np

class ColorAppLab:
    def __init__(self, root=None, model_path="saved_model.pth", img_size=(420, 420)):
        self.model_path = model_path
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])

        self.model = SimpleColorPredictor()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        self.max_display_width = 400
        self.padding_below_image = 80

        if root:
            self.root = root
            self.root.title("Dominujący kolor (Lab) - Predykcja")
            self.build_gui()

    def build_gui(self):
        self.btn = tk.Button(self.root, text="Wybierz obraz", command=self.load_image)
        self.btn.pack(pady=10)

        self.canvas = tk.Canvas(self.root)
        self.canvas.pack()

    def predict_dominant_color(self, image_path):
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)

        l, a, b = output[0][0]*100, output[0][1]*255 - 128, output[0][2]*255 - 128
        lab = np.array([l, a, b]).reshape(1, 1, 3)
        rgb = color.lab2rgb(lab).reshape(3)
        rgb = np.clip(rgb, 0, 1)
        rgb_int = [int(c * 255) for c in rgb]
        return rgb_int

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

        square_size = 50
        gap = 20
        text_gap = 30
        total_height = img_h + gap + square_size + text_gap + 20

        self.canvas.config(width=img_w, height=total_height)
        self.canvas.delete("all")

        x = img_w // 2
        self.canvas.create_image(x, 0, anchor=tk.N, image=self.tk_image)

        rgb = self.predict_dominant_color(file_path)
        hex_color = '#%02x%02x%02x' % tuple(rgb)

        square_x = x - square_size // 2
        square_y = img_h + gap

        self.canvas.create_rectangle(
            square_x, square_y,
            square_x + square_size, square_y + square_size,
            fill=hex_color, outline=""
        )

        self.canvas.create_text(
            x,
            square_y + square_size + text_gap,
            text=f"Dominujący kolor: RGB{tuple(rgb)}",
            font=("Arial", 11)
        )

    def run(self):
        self.root.mainloop()
