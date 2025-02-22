import torch
import torchvision.transforms as transforms
from PIL import Image

def load_image(image_path, shape=None):
    image = Image.open(image_path).convert("RGB")

    transform_list = [transforms.ToTensor()]
    
    if shape:
        shape = (int(shape[0]), int(shape[1]))  # Ensure shape values are integers
        transform_list.insert(0, transforms.Resize(shape))  # Resize before converting to tensor

    transform = transforms.Compose(transform_list)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    return image


def save_image(tensor, filename):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone().detach().squeeze(0)
    image = unloader(image)
    image.save(filename)
