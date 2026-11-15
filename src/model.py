import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(
    num_classes: int = 3,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Create a ResNet-18 classifier for three ultrasound classes."""

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    input_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(input_features, num_classes),
    )

    return model
