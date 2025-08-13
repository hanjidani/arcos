import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

class FeatureExtractor(nn.Module):
    """
    A simple CNN to extract features from MNIST images.
    Output will be a 128-dimensional feature vector.
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

def evaluate_model(model, classifier, data_loader, device):
    """Evaluates the model's accuracy on a given dataset."""
    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            features = model(data)
            outputs = classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def train_feature_extractor(model, train_loader, device, epochs=10):
    """
    Train the feature extractor model.
    """
    model.train()
    # Note: We add a classifier head here just for training purposes.
    # It will be discarded when we use the model as a feature extractor.
    classifier = nn.Linear(128, 10).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            features = model(data)
            output = classifier(features)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    print("Feature extractor training complete.")
    return model, classifier

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Model
    feature_extractor = FeatureExtractor().to(device)
    
    # Train
    trained_model, trained_classifier = train_feature_extractor(feature_extractor, train_loader, device, epochs=10)

    # Evaluate
    print("Evaluating on training data...")
    train_accuracy = evaluate_model(trained_model, trained_classifier, train_loader, device)
    print(f"Training Accuracy: {train_accuracy:.2f}%")

    print("Evaluating on test data...")
    test_accuracy = evaluate_model(trained_model, trained_classifier, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Save the trained model
    torch.save(trained_model.state_dict(), 'feature_extractor.pth')
    print("Trained feature extractor model saved to feature_extractor.pth")

    # Save results to CSV
    results = {
        'train_accuracy': [train_accuracy],
        'test_accuracy': [test_accuracy]
    }
    df = pd.DataFrame(results)
    
    # Ensure the results directory exists
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    df.to_csv(os.path.join(output_dir, 'feature_extractor_accuracy.csv'), index=False)
    print(f"Accuracy results saved to {os.path.join(output_dir, 'feature_extractor_accuracy.csv')}")
