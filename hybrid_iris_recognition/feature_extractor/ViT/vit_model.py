import torch
from torchvision import transforms
import timm

def get_vit_model(device):
    """"
    Returns a pre-trained Vision Transformer (ViT) model without the classification head.
    """
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.reset_classifier(0)
    model.eval()
    return model.to(device)

def get_vit_preprocess():
    """
    Returns the preprocessing transformations for ViT model.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])