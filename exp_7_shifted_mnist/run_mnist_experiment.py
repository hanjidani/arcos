import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import os
from tqdm import tqdm
from torchvision import datasets, transforms

from data_loader import get_mnist_loaders, apply_shift_to_dataset
from models import create_model

# --- Core Helper Functions ---

def train_model(model, train_loader, device, epochs=5):
    """
    Trains a model on the provided data loader.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model

def calculate_risk(model, data_loader, device):
    """
    Calculates the classification error (risk) of a model.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 1 - (correct / total)

def get_features(model, data_loader, device):
    """
    Extracts and returns flattened features from a model for all data in the loader.
    """
    model.eval()
    all_features = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            _, features = model(data, return_features=True)
            all_features.append(features.cpu())
    return torch.cat(all_features, dim=0)

def calculate_mmd(X, Y, gamma=None):
    """
    Calculates the empirical MMD between two sets of features.
    Uses the median heuristic to select gamma if not provided.
    """
    # Detach tensors from the computation graph
    X = X.detach()
    Y = Y.detach()

    # Gaussian RBF kernel
    K_XX = torch.exp(-torch.cdist(X, X)**2 / (2 * gamma**2))
    K_YY = torch.exp(-torch.cdist(Y, Y)**2 / (2 * gamma**2))
    K_XY = torch.exp(-torch.cdist(X, Y)**2 / (2 * gamma**2))

    mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return torch.sqrt(mmd2) if mmd2 > 0 else 0.0

# --- Main Experiment ---

def run_single_experiment(args):
    """
    Runs a single experiment for one model capacity and one shift severity.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Running experiment for: {args.shift_type}, capacity={args.model_capacity}, severity={args.shift_severity}")

    # 1. Load Data
    train_dataset = datasets.MNIST('../data', train=True, download=True)
    test_dataset = datasets.MNIST('../data', train=False, download=True)
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=1000
    )

    # 2. Train Source Model (Q)
    print(f"\n===== Training Model with {args.model_capacity} layers =====")
    model_q = create_model(args.model_capacity, device)
    train_model(model_q, train_loader, device, epochs=args.epochs)
    
    emp_risk_q_s = calculate_risk(model_q, train_loader, device)
    risk_q_p = calculate_risk(model_q, test_loader, device)
    g_q = np.abs(risk_q_p - emp_risk_q_s)

    # 3. Apply Shift
    shift_params = {}
    if args.shift_type == 'translation':
        shift_params = {'dx': int(args.shift_severity), 'dy': int(args.shift_severity)}
    elif args.shift_type == 'rotation':
        shift_params = {'angle': args.shift_severity}
    
    shifted_train_dataset = apply_shift_to_dataset(train_dataset, args.shift_type, shift_params)
    shifted_test_dataset = apply_shift_to_dataset(test_dataset, args.shift_type, shift_params)
    
    shifted_train_loader = torch.utils.data.DataLoader(shifted_train_dataset, batch_size=args.batch_size)
    shifted_test_loader = torch.utils.data.DataLoader(shifted_test_dataset, batch_size=1000)

    # 4. Train Target Model (Q_tilde)
    print(f"--- Training Target Model for severity {args.shift_severity} ---")
    model_q_tilde = create_model(args.model_capacity, device)
    train_model(model_q_tilde, shifted_train_loader, device, epochs=args.epochs)

    # 5. Calculate All Metrics
    risk_q_p_tilde = calculate_risk(model_q, shifted_test_loader, device)
    delta_r = np.abs(risk_q_p - risk_q_p_tilde)
    
    emp_risk_q_tilde_s_tilde = calculate_risk(model_q_tilde, shifted_train_loader, device)
    risk_q_tilde_p_tilde = calculate_risk(model_q_tilde, shifted_test_loader, device)
    g_q_tilde = np.abs(risk_q_tilde_p_tilde - emp_risk_q_tilde_s_tilde)

    emp_risk_q_s_tilde = calculate_risk(model_q, shifted_train_loader, device)
    d_q_q_tilde = np.abs(emp_risk_q_s_tilde - emp_risk_q_tilde_s_tilde)

    source_features = get_features(model_q, test_loader, device)
    target_features = get_features(model_q, shifted_test_loader, device)
    
    dists = torch.cdist(source_features, source_features)
    gamma = torch.median(dists[dists>0])
    
    mmd_dist = calculate_mmd(source_features, target_features, gamma=gamma).item()
    
    bound = g_q + g_q_tilde + d_q_q_tilde + mmd_dist
    tightness = bound / delta_r if delta_r > 0 else np.inf

    result = {
        'num_layers': args.model_capacity,
        'shift_severity': args.shift_severity,
        'shift_type': args.shift_type,
        'delta_r': delta_r,
        'g_q': g_q,
        'g_q_tilde': g_q_tilde,
        'd_q_q_tilde': d_q_q_tilde,
        'mmd_dist': mmd_dist,
        'bound': bound,
        'tightness_ratio': tightness
    }

    # 6. Save Single-Row Result
    df = pd.DataFrame([result])
    output_dir = os.path.join('exp_7_shifted_mnist', 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{args.shift_type}_{args.model_capacity}_{args.shift_severity}.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResult saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Single Shifted MNIST Experiment')
    parser.add_argument('--shift_type', type=str, required=True, choices=['translation', 'rotation'],
                        help='Type of shift to apply.')
    parser.add_argument('--model_capacity', type=int, required=True,
                        help='Number of model layers to test.')
    parser.add_argument('--shift_severity', type=float, required=True,
                        help='Severity of the shift to apply.')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to train models.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training.')
                        
    args = parser.parse_args()
    run_single_experiment(args)
