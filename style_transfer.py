import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Function to load an image
def load_image(image_path, size=512):
    image = Image.open(image_path)  # Open image
    transform = transforms.Compose([
        transforms.Resize((size, size)),  # Resize image
        transforms.ToTensor()  # Convert to tensor
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Load content and style images
content_img = load_image("content.jpg")
style_img = load_image("style.jpg")

# Show images
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(content_img.squeeze().permute(1,2,0))  # Convert tensor to image format
plt.title("Content Image")

plt.subplot(1,2,2)
plt.imshow(style_img.squeeze().permute(1,2,0))
plt.title("Style Image")

plt.show()
