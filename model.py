import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()  # Ensure eval mode
        
        # Select layers that capture content and style information
        self.selected_layers = {
            '0': 'conv1_1',   # First convolutional layer (style)
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # Used for content loss
            '28': 'conv5_1'   # Style layer
        }

        self.layers = nn.ModuleDict({name: vgg[int(layer)] for layer, name in self.selected_layers.items()})

        # Freeze VGG model parameters (no training required)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Extracts features from selected layers."""
        features = {}
        for name, layer in self.layers.items():
            x = layer(x)
            features[name] = x  # Store feature maps
        return features

# ✅ **Fixed: Ensure correct tensor extraction in loss calculations**
def content_loss(generated_features, content_features):
    """Computes content loss between generated and content images."""
    return F.mse_loss(generated_features['conv4_2'], content_features['conv4_2'].detach())

def gram_matrix(tensor):
    """Computes the Gram matrix for style loss calculation."""
    _, channels, height, width = tensor.size()
    tensor = tensor.view(channels, height * width)  # Flatten feature map
    return torch.mm(tensor, tensor.t()) / (channels * height * width)

def style_loss(generated_features, style_features):
    """Calculates style loss using Gram matrices."""
    loss = 0
    for layer in generated_features:
        gen_gram = gram_matrix(generated_features[layer])
        style_gram = gram_matrix(style_features[layer])
        loss += F.mse_loss(gen_gram, style_gram.detach())  # Ensure tensors match
    return loss
