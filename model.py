import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_model(num_classes):
    """
    Returns a ResNet50 model modified for multi-label classification.
    """
    # Load pretrained model
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    
    # Replace the final fully connected layer
    # ResNet50's final layer is named 'fc' and has 2048 input features
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

if __name__ == '__main__':
    model = get_model(14)
    print("Model created.")
    # shape check
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(f"Output shape: {y.shape}")
    assert y.shape == (1, 14)
    print("Model test passed.")
