import os
import torch
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from torchvision import transforms
from Files.model import SimpleColorPredictor


class ColorApp:
    def __init__(self, root, model_dir=".", img_size=(420, 420), max_colors=5):
        self.root = root
        self.root.title("Color predictor")

        self.img_size = img_size
        self.model_dir = model_dir
        self.max_colors = max_colors
        self.selected_colors = 1
        self.image_path = None
        self.model = None

        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])

        self.build_gui()

    def build_gui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=(20, 10))  #

        self.btn_choose = tk.Button(top_frame, text="Choose picture", font=("Arial", 11), command=self.choose_image)
        self.btn_choose.pack(side=tk.LEFT, padx=8)

        self.combo_colors = ttk.Combobox(top_frame, values=[str(i) for i in range(1, self.max_colors + 1)],
                                         width=5, state="readonly", font=("Arial", 11))
        self.combo_colors.current(0)
        self.combo_colors.pack(side=tk.LEFT, padx=8)

        self.btn_predict = tk.Button(top_frame, text="Predicate", font=("Arial", 11),
                                     command=self.run_prediction)
        self.btn_predict.pack(side=tk.LEFT, padx=8)

        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=15, pady=(10, 20))

        self.image_canvas = tk.Canvas(main_frame, width=600, height=600)
        self.image_canvas.pack(side=tk.LEFT, pady=(0, 0))

        self.color_frame = tk.Frame(main_frame)
        self.color_frame.pack(side=tk.LEFT, padx=25)

    def choose_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Obrazy", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.image_path = file_path
            image = Image.open(file_path).convert("RGB")
            resized = self.resize_to_fit(image, 600)
            self.tk_image = ImageTk.PhotoImage(resized)

            self.image_canvas.config(width=resized.width, height=resized.height)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def run_prediction(self):
        if not self.image_path:
            return

        try:
            self.selected_colors = int(self.combo_colors.get())
        except ValueError:
            self.selected_colors = 1

        model_path = os.path.join(self.model_dir, f"saved_model_{self.selected_colors}.pth")
        if not os.path.exists(model_path):
            print("Model nie istnieje:", model_path)
            return

        self.model = SimpleColorPredictor(num_colors=self.selected_colors)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        image = Image.open(self.image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)[0]

        for widget in self.color_frame.winfo_children():
            widget.destroy()

        for i, rgb in enumerate(output):
            r, g, b = [int(c.item() * 255) for c in rgb]
            hex_color = '#%02x%02x%02x' % (r, g, b)

            canvas = tk.Canvas(self.color_frame, width=80, height=40, bg=hex_color, highlightthickness=1)
            canvas.pack(pady=(10 if i == 0 else 6, 2))  # Kolorowy prostokąt pierwszy

            label = tk.Label(self.color_frame, text=f"RGB({r}, {g}, {b})", bg="white", font=("Arial", 12))
            label.pack(pady=(0, 6))

    def resize_to_fit(self, image, max_width):
        img_w, img_h = image.size
        if img_w > max_width:
            ratio = max_width / img_w
            new_size = (int(img_w * ratio), int(img_h * ratio))
            return image.resize(new_size, Image.Resampling.LANCZOS)
        return image
