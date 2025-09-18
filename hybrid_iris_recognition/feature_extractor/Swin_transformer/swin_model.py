import torch
import torchvision.transforms as transforms
from torchvision.models import swin_t, Swin_T_Weights

def get_swin_model(device):
    """
    Returns a Swin Transformer model pre-trained on ImageNet-1K.
    """
    weights = Swin_T_Weights.DEFAULT
    model = swin_t(weights=weights)
    model.eval()
    model.head = torch.nn.Identity()
    return model.to(device)

def get_swin_preprocess():
    """
    Returns the preprocessing transforms for the Swin Transformer model.
    """
    weights = Swin_T_Weights.DEFAULT
    preprocess = weights.transforms()
    return preprocess
