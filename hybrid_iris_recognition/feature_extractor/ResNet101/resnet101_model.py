import torch
import torchvision.transforms as transforms
from torchvision.models import resnet101, ResNet101_Weights


def get_resnet_model(device):
    """
    Returns a pretrained ResNet101 model without the final classification layer.
    """
    model = resnet101(weights=ResNet101_Weights.DEFAULT)
    model.eval()
    model = torch.nn.Sequential(*list(model.children())[:-1])
    return model.to(device)

def get_resnet_preprocess():
    """
    Returns the preprocessing pipeline required for ResNet101 input.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])