import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU()
        )
        self.final_fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64 * 8 * 8) # Flatten
        x = self.classifier(x)
        x = self.final_fc(x)
        return x

def get_cnn_model(num_classes=5):
    """
    Returns an instance of the SimpleCNN model.
    """
    return SimpleCNN(num_classes=num_classes)

def get_feature_extractor(model):
    """
    Returns the feature extractor part of the model 
    (all layers up to the penultimate layer's activations).
    """
    return nn.Sequential(
        model.features,
        nn.Flatten(),
        model.classifier
    )
