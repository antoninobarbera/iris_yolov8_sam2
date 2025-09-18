import torch
import torchvision.transforms as transforms
from torchvision.models import densenet121, DenseNet121_Weights

def get_densenet_model(device):
    """
    Returns a pretrained DenseNet121 model without the final classification layer.
    """
    weights = DenseNet121_Weights.DEFAULT
    model = densenet121(weights=weights)
    model.eval()

    class DenseNetFeatureExtractor(torch.nn.Module):
        def __init__(self, densenet):
            super().__init__()
            self.features = densenet.features
            self.pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        def forward(self, x):
            x = self.features(x)
            x = torch.relu(x)
            x = self.pooling(x)
            return x

    feature_extractor = DenseNetFeatureExtractor(model)
    return feature_extractor.to(device)


def get_densenet_preprocess():
    """
    Returns the preprocessing pipeline required for DenseNet121 input.
    """
    weights = DenseNet121_Weights.DEFAULT
    preprocess = weights.transforms()
    return preprocess