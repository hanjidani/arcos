import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import pandas as pd
import numpy as np

from data_loader import get_dataloaders
from models import get_cnn_model, get_feature_extractor
from src.wasserstein import calculate_wasserstein_nd, calculate_mmd
from src.lipschitz import get_lipschitz_upper_bound

def train_model(model, dataloader, num_epochs=10, lr=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"  Epoch {epoch+1}/{num_epochs} completed.")
    return model

def evaluate_model(model, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)

def get_features(model, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_extractor = get_feature_extractor(model)
    feature_extractor.to(device)
    feature_extractor.eval()
    
    features_list = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            features = feature_extractor(inputs)
            features_list.append(features.cpu().numpy())
    return np.concatenate(features_list)

def main(args):
    results_list = []
    corruption_types = ['noise', 'blur', 'brightness', 'jpeg']
    severities = [0.1, 0.3, 0.5, 0.7, 0.9]

    for corruption in corruption_types:
        for severity in severities:
            print(f"\n--- Running experiment: Corruption={corruption}, Severity={severity} ---")
            
            # 1. Get Dataloaders
            clean_loader, corrupted_loader = get_dataloaders(corruption, severity, args.batch_size)
            
            # 2. Train Models
            print("Training model Q on clean data...")
            q_model = get_cnn_model()
            q_model = train_model(q_model, clean_loader, num_epochs=args.num_epochs, lr=args.lr)
            
            print("Training model Q_tilde on corrupted data...")
            q_tilde_model = get_cnn_model()
            q_tilde_model = train_model(q_tilde_model, corrupted_loader, num_epochs=args.num_epochs, lr=args.lr)

            # 3. ARCOS Decomposition
            print("Performing ARCOS decomposition...")
            
            # Risks for delta_r
            r_tilde_q = evaluate_model(q_model, corrupted_loader)
            r_tilde_q_tilde = evaluate_model(q_tilde_model, corrupted_loader)
            delta_r = abs(r_tilde_q - r_tilde_q_tilde)
            
            # Generalization Gaps
            g_q = evaluate_model(q_model, clean_loader)
            g_q_tilde = r_tilde_q_tilde
            
            # Divergence Terms
            features_clean = get_features(q_model, clean_loader)
            features_corrupted = get_features(q_model, corrupted_loader)
            w1_dist = calculate_wasserstein_nd(features_clean, features_corrupted)
            mmd_dist = calculate_mmd(features_clean, features_corrupted)
            
            # Output Distance
            features_q_corrupted = get_features(q_model, corrupted_loader)
            features_q_tilde_corrupted = get_features(q_tilde_model, corrupted_loader)
            output_dist = np.linalg.norm(features_q_corrupted - features_q_tilde_corrupted, axis=1).mean()
            
            # Lipschitz Constant
            lx = get_lipschitz_upper_bound(get_feature_extractor(q_tilde_model))
            
            # Full Bound Calculation (L_ell for CrossEntropy is typically <= 1)
            l_ell = 1.0 
            bound = g_q + g_q_tilde + (lx * w1_dist) + (l_ell * output_dist)
            
            # Record Results
            results_list.append({
                'corruption': corruption,
                'severity': severity,
                'delta_r': delta_r,
                'arcos_bound': bound,
                'tightness_ratio': delta_r / bound if bound > 0 else 0,
                'g_q': g_q,
                'g_q_tilde': g_q_tilde,
                'wasserstein_divergence': w1_dist,
                'mmd_divergence': mmd_dist,
                'output_distance': output_dist,
                'lipschitz_Lx': lx
            })

    # Save final results
    df = pd.DataFrame(results_list)
    df.to_csv('arcos_corruption_results.csv', index=False)
    print("\n--- Experiment Complete ---")
    print("Results saved to arcos_corruption_results.csv")
    print(df.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARCOS Corruption experiment on CIFAR-10.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for models.')
    args = parser.parse_args()
    main(args)
