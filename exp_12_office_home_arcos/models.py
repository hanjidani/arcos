import torch
import torchvision.models as models

def get_resnet_model(num_classes=65, pretrained=True):
    """
    Returns a pre-trained ResNet-18 model with the final layer modified for the given number of classes.
    """
    model = models.resnet18(pretrained=pretrained)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    return model

def get_feature_extractor(model):
    """
    Returns the feature extractor part of the model (all layers except the final fully connected layer).
    """
    return torch.nn.Sequential(*list(model.children())[:-1])

if __name__ == '__main__':
    # Test the model and feature extractor
    model = get_resnet_model()
    feature_extractor = get_feature_extractor(model)
    
    print("Full Model:")
    print(model)
    
    print("\nFeature Extractor:")
    print(feature_extractor)
    
    # Test with a dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    features = feature_extractor(dummy_input)
    
    print(f"\nDummy input shape: {dummy_input.shape}")
    print(f"Model output shape: {output.shape}")
    print(f"Feature extractor output shape: {features.shape}")
