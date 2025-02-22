import torch
import torch.optim as optim
from model import VGGFeatures, gram_matrix
from utils import load_image, save_image

# Device configuration (Runs on GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load images (Ensure correct resizing)
content_image = load_image("content.jpg").to(device)
style_image = load_image("style.jpg", shape=(content_image.shape[2], content_image.shape[3])).to(device)

# Load model
vgg = VGGFeatures().to(device).eval()

# Extract features from images
content_features = vgg(content_image)
style_features = vgg(style_image)

# Initialize generated image with content image (for better convergence)
generated_image = content_image.clone().detach().requires_grad_(True)

# Define optimizer
optimizer = optim.Adam([generated_image], lr=0.01)

# **Balanced Loss Weights** (Avoid extreme values)
content_weight = 1e4  # Tuned for stability
style_weight = 1e6  # Tuned to balance content & style

# Training loop
num_steps = 500
for step in range(num_steps):
    optimizer.zero_grad()
    
    generated_features = vgg(generated_image)

    # Compute content loss
    content_loss = torch.nn.functional.mse_loss(
        generated_features['conv4_2'], content_features['conv4_2']
    )

    # Compute style loss
    style_loss = 0
    for layer in style_features:
        gen_gram = gram_matrix(generated_features[layer])
        style_gram = gram_matrix(style_features[layer])
        style_loss += torch.nn.functional.mse_loss(gen_gram, style_gram)

    # **Stable Total Loss**
    total_loss = content_weight * content_loss + style_weight * style_loss
    total_loss.backward()

    optimizer.step()

    # Print loss updates
    if step % 50 == 0:
        print(f"Step {step}: Loss = {total_loss.item():.2f} (Content: {content_loss.item():.2f}, Style: {style_loss.item():.2f})")

# Save final output
save_image(generated_image, "output.jpg")
print("✅ Style transfer complete! 🎨 Image saved as output.jpg.")
