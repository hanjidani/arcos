import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import os
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

# Adjust import paths for the new directory structure
from exp_9_feature_space_shifts.feature_extractor import FeatureExtractor
from adversarial_utils import find_worst_case_shift, calculate_wasserstein1_distance, calculate_output_distance, calculate_risk_linear

# --- Model Definitions ---

class LinearClassifier(nn.Module):
    """A simple linear classifier."""
    def __init__(self, feature_dim=128, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# --- Helper Functions (some copied from exp_9) ---

def get_all_features_and_labels(feature_extractor, data_loader, device):
    """Extracts features and labels for the entire dataset."""
    feature_extractor.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for data, labels in tqdm(data_loader, desc="Extracting features"):
            data = data.to(device)
            features = feature_extractor(data)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_features), torch.cat(all_labels)

def train_linear_model(model, features, labels, device, epochs=50):
    """Trains a linear classifier on top of the extracted features."""
    model.train()
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for feature_batch, label_batch in loader:
            feature_batch, label_batch = feature_batch.to(device), label_batch.to(device)
            optimizer.zero_grad()
            output = model(feature_batch)
            loss = criterion(output, label_batch)
            loss.backward()
            optimizer.step()
    return model

# --- Main Experiment ---

def run_adversarial_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Pretrained Feature Extractor
    # Adjust path to be relative to the project root
    feature_extractor_path = os.path.join('exp_9_feature_space_shifts', 'feature_extractor.pth')
    feature_extractor = FeatureExtractor()
    try:
        feature_extractor.load_state_dict(torch.load(feature_extractor_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Pretrained feature extractor not found at {feature_extractor_path}")
        print("Please run `python exp_9_feature_space_shifts/feature_extractor.py` first.")
        return
    feature_extractor.to(device)
    feature_extractor.eval()

    # 2. Load Data and Extract Features
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    full_test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(full_train_dataset, batch_size=1024)
    test_loader = DataLoader(full_test_dataset, batch_size=1024)

    source_train_features, source_train_labels = get_all_features_and_labels(feature_extractor, train_loader, device)
    source_test_features, source_test_labels = get_all_features_and_labels(feature_extractor, test_loader, device)
    
    # 3. Train Source Classifier (Q)
    print("\n===== Training Source Classifier (Q) =====")
    model_q = LinearClassifier().to(device)
    train_linear_model(model_q, source_train_features, source_train_labels, device, epochs=args.epochs)

    # For this experiment, we assume Q_tilde is a slightly perturbed version of Q
    # or a model trained on slightly different data. For simplicity, we can just use
    # a re-initialized model or the same model. The bound should still hold.
    # We will train it on the same source data for this example.
    print("\n===== Training Target Classifier (Q_tilde) =====")
    model_q_tilde = LinearClassifier().to(device)
    train_linear_model(model_q_tilde, source_train_features, source_train_labels, device, epochs=args.epochs)

    # 4. Find Worst-Case Shift
    print("\n===== Searching for Adversarial Shift =====")
    worst_case_result = find_worst_case_shift(
        model_q, model_q_tilde, feature_extractor, 
        source_test_features, source_test_labels,
        test_loader, # Pass the test loader for Lipschitz estimation
        device, 
        budget=args.search_budget, 
        shift_magnitude=args.shift_magnitude
    )

    # 5. Save Result
    if worst_case_result:
        df = pd.DataFrame([worst_case_result])
        output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'adversarial_tightness_results.csv')
        df.to_csv(output_path, index=False)
        print(f"\nWorst-case shift results saved to {output_path}")
        print(df)
    else:
        print("\nCould not find a valid adversarial shift.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Adversarial Tightness Experiment')
    parser.add_argument('--search_budget', type=int, default=1000,
                        help='Number of iterations for the adversarial search.')
    parser.add_argument('--shift_magnitude', type=float, default=0.5,
                        help='Magnitude of the random shift to apply.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train linear classifiers.')
                        
    args = parser.parse_args()
    run_adversarial_experiment(args)
