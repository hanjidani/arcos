import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Base CNN Block ---

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))

# --- Dynamic CNN Model ---

class DynamicCNN(nn.Module):
    def __init__(self, num_layers=4, num_classes=10):
        """
        Creates a CNN with a variable number of convolutional layers.
        
        Args:
            num_layers (int): The number of ConvBlocks to include (e.g., 2, 4, 6, 8).
            num_classes (int): The number of output classes.
        """
        super(DynamicCNN, self).__init__()
        if num_layers not in [2, 4, 6, 8]:
            raise ValueError("Number of layers must be 2, 4, 6, or 8.")
            
        layers = []
        in_channels = 1
        out_channels = 16
        
        for i in range(num_layers):
            layers.append(ConvBlock(in_channels, out_channels))
            in_channels = out_channels
            if (i + 1) % 2 == 0:
                out_channels *= 2 # Double channels every 2 layers
        
        self.features = nn.Sequential(*layers)
        
        # Calculate the flattened feature size dynamically
        # The input image is 28x28. It gets pooled 'num_layers' times.
        final_size = 28 // (2**num_layers)
        self.fc_in_features = in_channels * final_size * final_size
        
        self.classifier = nn.Linear(self.fc_in_features, num_classes)

    def forward(self, x, return_features=False):
        features = self.features(x)
        flat_features = features.view(-1, self.fc_in_features)
        output = self.classifier(flat_features)
        if return_features:
            return output, flat_features
        return output

# --- Helper function to create models ---

def create_model(num_layers, device):
    """
    Initializes a DynamicCNN model and moves it to the specified device.
    """
    model = DynamicCNN(num_layers=num_layers)
    model.to(device)
    return model

if __name__ == '__main__':
    # --- Example Usage ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create a 2-layer model
    print("\n--- Model with 2 layers ---")
    model_2 = create_model(2, device)
    print(model_2)
    print(f"Number of parameters: {sum(p.numel() for p in model_2.parameters())}")


    # Create a 4-layer model
    print("\n--- Model with 4 layers ---")
    model_4 = create_model(4, device)
    print(model_4)
    print(f"Number of parameters: {sum(p.numel() for p in model_4.parameters())}")

    # Create an 8-layer model
    print("\n--- Model with 8 layers ---")
    model_8 = create_model(8, device)
    print(model_8)
    print(f"Number of parameters: {sum(p.numel() for p in model_8.parameters())}")

    # Test forward pass with dummy data
    dummy_input = torch.randn(64, 1, 28, 28).to(device)
    output = model_4(dummy_input)
    print(f"\nOutput shape for 4-layer model: {output.shape}") # Should be [64, 10]
