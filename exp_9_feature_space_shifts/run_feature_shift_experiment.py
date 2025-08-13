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

from feature_extractor import FeatureExtractor

# --- Model Definitions ---

class LinearClassifier(nn.Module):
    """A simple linear classifier."""
    def __init__(self, feature_dim=128, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# --- Helper Functions ---

def calculate_risk_linear(model, features, labels, device):
    """
    Calculates the classification error (risk) of a linear model on features.
    """
    model.eval()
    model.to(device)
    features = features.to(device)
    labels = labels.to(device)
    
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=1024)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for feature_batch, label_batch in loader:
            outputs = model(feature_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += label_batch.size(0)
            correct += (predicted == label_batch).sum().item()
    if total == 0:
        return 0
    return 1 - (correct / total)


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

def calculate_mmd(X, Y, gamma):
    """
    Calculates the empirical MMD between two sets of features.
    """
    X = X.detach()
    Y = Y.detach()

    K_XX = torch.exp(-torch.cdist(X, X)**2 / (2 * gamma**2))
    K_YY = torch.exp(-torch.cdist(Y, Y)**2 / (2 * gamma**2))
    K_XY = torch.exp(-torch.cdist(X, Y)**2 / (2 * gamma**2))

    mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return torch.sqrt(mmd2) if mmd2 > 0 else 0.0

# --- Shift Functions ---

def apply_subspace_shift(features, severity=0.5):
    """
    Applies a shift by projecting features away from a random subspace.
    Severity controls the dimension of the subspace to project away.
    """
    print(f"Applying subspace shift with severity {severity}...")
    n_samples, n_features = features.shape
    
    # Number of dimensions to project away
    n_dims_to_remove = int(n_features * severity)
    if n_dims_to_remove == 0:
        return features
    if n_dims_to_remove >= n_features:
        return torch.zeros_like(features) # Remove all dimensions

    # Create a random projection matrix
    projection_matrix, _ = torch.linalg.qr(torch.randn(n_features, n_features))
    
    # Keep the first (n_features - n_dims_to_remove) components
    subspace_basis = projection_matrix[:, :-n_dims_to_remove]
    
    # Project and reconstruct
    projected_features = features @ subspace_basis
    reconstructed_features = projected_features @ subspace_basis.T
    
    return reconstructed_features

# --- Main Experiment ---

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Pretrained Feature Extractor
    feature_extractor = FeatureExtractor()
    try:
        feature_extractor.load_state_dict(torch.load(args.feature_extractor_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Pretrained feature extractor not found at {args.feature_extractor_path}")
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
    
    # Use smaller loaders for faster feature extraction
    train_loader = DataLoader(full_train_dataset, batch_size=1024)
    test_loader = DataLoader(full_test_dataset, batch_size=1024)

    source_train_features, source_train_labels = get_all_features_and_labels(feature_extractor, train_loader, device)
    source_test_features, source_test_labels = get_all_features_and_labels(feature_extractor, test_loader, device)
    
    # 3. Train Source Classifier (Q)
    print("\n===== Training Source Classifier (Q) =====")
    model_q = LinearClassifier().to(device)
    train_linear_model(model_q, source_train_features, source_train_labels, device, epochs=args.epochs)

    # 4. Apply Shift to Features
    target_train_features = apply_subspace_shift(source_train_features, args.shift_severity)
    target_test_features = apply_subspace_shift(source_test_features, args.shift_severity)
    
    # 5. Train Target Classifier (Q_tilde)
    print("\n===== Training Target Classifier (Q_tilde) =====")
    model_q_tilde = LinearClassifier().to(device)
    train_linear_model(model_q_tilde, target_train_features, source_train_labels, device, epochs=args.epochs) # Labels remain the same

    # 6. Calculate All Metrics
    print("\n===== Calculating Metrics =====")
    
    # Risks on test sets
    risk_q_p = calculate_risk_linear(model_q, source_test_features, source_test_labels, device)
    risk_q_p_tilde = calculate_risk_linear(model_q, target_test_features, source_test_labels, device) # Source model on shifted data
    risk_q_tilde_p_tilde = calculate_risk_linear(model_q_tilde, target_test_features, source_test_labels, device)
    
    # Delta R (the ground truth we want to bound)
    delta_r = np.abs(risk_q_p - risk_q_p_tilde)
    
    # Generalization Gaps
    emp_risk_q_s = calculate_risk_linear(model_q, source_train_features, source_train_labels, device)
    g_q = np.abs(risk_q_p - emp_risk_q_s)
    
    emp_risk_q_tilde_s_tilde = calculate_risk_linear(model_q_tilde, target_train_features, source_train_labels, device)
    g_q_tilde = np.abs(risk_q_tilde_p_tilde - emp_risk_q_tilde_s_tilde)
    
    # Discrepancy Term
    emp_risk_q_s_tilde = calculate_risk_linear(model_q, target_train_features, source_train_labels, device)
    d_q_q_tilde = np.abs(emp_risk_q_s_tilde - emp_risk_q_tilde_s_tilde)

    # MMD Calculation
    dists = torch.cdist(source_test_features, source_test_features)
    gamma = torch.median(dists[dists > 0])
    mmd_dist = calculate_mmd(source_test_features, target_test_features, gamma).item()
    
    # Final Bound
    bound = g_q + g_q_tilde + d_q_q_tilde + mmd_dist
    tightness = bound / delta_r if delta_r > 0 else np.inf

    result = {
        'shift_type': 'subspace',
        'shift_severity': args.shift_severity,
        'delta_r': delta_r,
        'g_q': g_q,
        'g_q_tilde': g_q_tilde,
        'd_q_q_tilde': d_q_q_tilde,
        'mmd_dist': mmd_dist,
        'bound': bound,
        'tightness_ratio': tightness
    }

    # 7. Save Result
    df = pd.DataFrame([result])
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'subspace_{args.shift_severity}.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResult saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Feature-Space Shift Experiment')
    parser.add_argument('--feature_extractor_path', type=str, default='feature_extractor.pth',
                        help='Path to the pretrained feature extractor model.')
    parser.add_argument('--shift_severity', type=float, default=0.5,
                        help='Severity of the feature-space shift to apply.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train linear classifiers.')
                        
    args = parser.parse_args()
    run_experiment(args)
