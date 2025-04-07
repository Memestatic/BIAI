import torch

def hex_to_rgb_tensor(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return torch.tensor([r, g, b])

def process_annotation(colors_hex_list):
    # Każdy element listy to tensor o wymiarach (3,)
    colors_tensors = [hex_to_rgb_tensor(color) for color in colors_hex_list]
    return torch.stack(colors_tensors)  # Tensor o wymiarach (N, 3)
