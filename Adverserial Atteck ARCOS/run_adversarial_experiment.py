import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import os
import random
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

# Fix import paths and add enhanced utilities
try:
    from exp_9_feature_space_shifts.feature_extractor import FeatureExtractor
except ImportError:
    # Fallback: create a simple feature extractor if the module is not available
    class FeatureExtractor(nn.Module):
        def __init__(self, feature_dim=128):
            super(FeatureExtractor, self).__init__()
            self.fc = nn.Linear(784, feature_dim)  # MNIST: 28x28 = 784
            
        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten
            return self.fc(x)

from adversarial_utils import (
    find_worst_case_shift, 
    calculate_wasserstein1_distance, 
    calculate_output_distance, 
    calculate_risk_linear,
    calculate_expected_loss,
    estimate_lipschitz_constant_loss_based,
    calculate_output_distance_kl
)

# --- Reproducibility Setup ---

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def train_linear_model(model, features, labels, device, epochs=50, val_split=0.2):
    """Trains a linear classifier on top of the extracted features with validation split."""
    model.train()
    
    # Create train/val split
    n_samples = len(features)
    n_val = int(n_samples * val_split)
    indices = torch.randperm(n_samples)
    
    train_features = features[indices[n_val:]]
    train_labels = labels[indices[n_val:]]
    val_features = features[indices[:n_val]]
    val_labels = labels[indices[:n_val]]
    
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for feature_batch, label_batch in train_loader:
            feature_batch, label_batch = feature_batch.to(device), label_batch.to(device)
            optimizer.zero_grad()
            output = model(feature_batch)
            loss = criterion(output, label_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for feature_batch, label_batch in val_loader:
                feature_batch, label_batch = feature_batch.to(device), label_batch.to(device)
                output = model(feature_batch)
                loss = criterion(output, label_batch)
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += label_batch.size(0)
                val_correct += (predicted == label_batch).sum().item()
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
    
    return model

# --- Main Experiment ---

def run_adversarial_experiment(args):
    # Set reproducibility with robust seeding
    set_seed(args.seed)
    
    # Additional reproducibility measures
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Seed: {args.seed}")

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

    # 4. Calculate Initial Metrics for Diagnostics
    print("\n===== Calculating Initial Metrics =====")
    
    # Risks and generalization gaps
    risk_q_p = calculate_risk_linear(model_q, source_test_features, source_test_labels, device)
    emp_risk_q_s = calculate_risk_linear(model_q, source_train_features, source_train_labels, device)
    g_q = np.abs(risk_q_p - emp_risk_q_s)
    
    risk_q_tilde_p = calculate_risk_linear(model_q_tilde, source_test_features, source_test_labels, device)
    emp_risk_q_tilde_s = calculate_risk_linear(model_q_tilde, source_train_features, source_train_labels, device)
    g_q_tilde = np.abs(risk_q_tilde_p - emp_risk_q_tilde_s)
    
    # Expected losses for bound tightness analysis
    expected_loss_q = calculate_expected_loss(model_q, source_test_features, source_test_labels, device)
    expected_loss_q_tilde = calculate_expected_loss(model_q_tilde, source_test_features, source_test_labels, device)
    
    # Lipschitz constants
    l_x_q = estimate_lipschitz_constant_loss_based(model_q, test_loader, device)
    l_x_q_tilde = estimate_lipschitz_constant_loss_based(model_q_tilde, test_loader, device)
    
    print(f"Initial Metrics:")
    print(f"  Risk Q on P: {risk_q_p:.4f}")
    print(f"  Risk Q_tilde on P: {risk_q_tilde_p:.4f}")
    print(f"  Gen gap Q: {g_q:.4f}")
    print(f"  Gen gap Q_tilde: {g_q_tilde:.4f}")
    print(f"  Expected loss Q: {expected_loss_q:.4f}")
    print(f"  Expected loss Q_tilde: {expected_loss_q_tilde:.4f}")
    print(f"  Lipschitz Q: {l_x_q:.4f}")
    print(f"  Lipschitz Q_tilde: {l_x_q_tilde:.4f}")

    # 5. Find Worst-Case Shift
    print("\n===== Searching for Adversarial Shift =====")
    
    # Create feature-space loader for Lipschitz estimation (not image-space loader)
    feat_ds = torch.utils.data.TensorDataset(source_test_features, source_test_labels)
    feat_loader = torch.utils.data.DataLoader(feat_ds, batch_size=1024)
    
    worst_case_result = find_worst_case_shift(
        model_q, model_q_tilde, feature_extractor, 
        source_test_features, source_test_labels,
        feat_loader, # Pass the feature-space loader for Lipschitz estimation
        device, 
        budget=args.search_budget, 
        shift_magnitude=args.shift_magnitude
    )

    # 6. Calculate Final Bound Components and Tightness
    if worst_case_result:
        print("\n===== Final Bound Analysis =====")
        
        # Calculate Wasserstein distance and output distance for the worst shift
        # (This would require recreating the shifted features, but for now we'll use the result)
        
        # Calculate tightness
        delta_r = worst_case_result.get('delta_r', 0)
        if delta_r > 0:
            # For demonstration, we'll use the bound components from the search
            bound_components = {
                'g_q': g_q,
                'g_q_tilde': g_q_tilde,
                'lipschitz_q': l_x_q,
                'lipschitz_q_tilde': l_x_q_tilde,
                'w1_dist': worst_case_result.get('w1_dist', 0),
                'output_dist': worst_case_result.get('output_dist', 0)
            }
            
                    # Add bound components to the result
        worst_case_result.update(bound_components)
        worst_case_result['tightness'] = worst_case_result.get('ratio', 0)
        
        # Add space information for clarity
        worst_case_result['w1_space'] = 'features'  # W1 computed on feature space
        worst_case_result['lipschitz_space'] = 'features'  # Lipschitz wrt feature space
        
        print(f"Bound Components:")
        for key, value in bound_components.items():
            print(f"  {key}: {value:.6f}")
        print(f"Tightness: {worst_case_result['tightness']:.4f}")
        print(f"Space: W1 on {worst_case_result['w1_space']}, Lipschitz wrt {worst_case_result['lipschitz_space']}")
        
        # 7. Save Enhanced Results
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
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
                        
    args = parser.parse_args()
    run_adversarial_experiment(args)
