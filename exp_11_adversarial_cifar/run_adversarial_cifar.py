import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import os
import torchattacks
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

# --- Standalone End-to-End Model for CIFAR-10 ---
class CifarModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc_layers(x)
        return x

# --- Helper Functions (same as before) ---

def train_model(model, loader, device, epochs, desc):
    model.to(device).train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for _ in tqdm(range(epochs), desc=desc):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
    return model

def calculate_risk(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 1 - (correct / total) if total > 0 else 0

def get_all_data(loader):
    all_images = torch.cat([data for data, _ in loader], 0)
    all_labels = torch.cat([labels for _, labels in loader], 0)
    return all_images, all_labels

def calculate_distance(X, Y, metric='w1'):
    X_flat = X.view(X.size(0), -1)
    Y_flat = Y.view(Y.size(0), -1)
    
    if len(X) > 1000: # Reduce sample size for faster distance calculation
        indices = np.random.choice(len(X), 1000, replace=False)
        X_flat, Y_flat = X_flat[indices], Y_flat[indices]

    if metric == 'w1':
        cost_matrix = torch.cdist(X_flat, Y_flat, p=2)
        return torch.mean(torch.min(cost_matrix, dim=1)[0]).item()
    elif metric == 'mmd':
        gamma = torch.median(torch.cdist(X_flat, X_flat)[X_flat.shape[0] > 1])
        K_XX = torch.exp(-torch.cdist(X_flat, X_flat)**2 / (2 * gamma**2))
        K_YY = torch.exp(-torch.cdist(Y_flat, Y_flat)**2 / (2 * gamma**2))
        K_XY = torch.exp(-torch.cdist(X_flat, Y_flat)**2 / (2 * gamma**2))
        mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
        return torch.sqrt(mmd2).item() if mmd2 > 0 else 0.0

# --- Main Experiment ---

def run_cifar_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load CIFAR-10 Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 3-channel normalization
    ])
    train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    source_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    source_test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # 2. Train Source Model (Q) from Scratch
    model_q = train_model(CifarModel(), source_train_loader, device, args.epochs, "Training source model Q")

    # 3. Generate Adversarial Dataset (P_tilde)
    print("\nCreating Adversarial Distribution P_tilde...")
    attack = torchattacks.PGD(model_q, eps=args.attack_eps, alpha=args.attack_alpha, steps=args.attack_steps)
    adv_train_images = torch.cat([attack(img.to(device), lbl.to(device)).cpu() for img, lbl in tqdm(source_train_loader, desc="Attacking train set")])
    adv_train_labels = get_all_data(source_train_loader)[1]
    adv_test_images = torch.cat([attack(img.to(device), lbl.to(device)).cpu() for img, lbl in tqdm(source_test_loader, desc="Attacking test set")])
    adv_test_labels = get_all_data(source_test_loader)[1]

    target_train_loader = DataLoader(TensorDataset(adv_train_images, adv_train_labels), batch_size=args.batch_size, shuffle=True)
    target_test_loader = DataLoader(TensorDataset(adv_test_images, adv_test_labels), batch_size=args.batch_size)

    # 4. Train Target Model (Q_tilde) from Scratch on Adversarial Data
    model_q_tilde = train_model(CifarModel(), target_train_loader, device, args.epochs, "Training target model Q_tilde")

    # 5. Calculate All Metrics for the Bound
    print("\nCalculating Final Metrics...")
    
    # Risks and Gaps
    risk_q_p = calculate_risk(model_q, source_test_loader, device)
    emp_risk_q_s = calculate_risk(model_q, source_train_loader, device)
    g_q = np.abs(risk_q_p - emp_risk_q_s)

    risk_q_tilde_p_tilde = calculate_risk(model_q_tilde, target_test_loader, device)
    emp_risk_q_tilde_s_tilde = calculate_risk(model_q_tilde, target_train_loader, device)
    g_q_tilde = np.abs(risk_q_tilde_p_tilde - emp_risk_q_tilde_s_tilde)
    
    d_q_q_tilde = np.abs(emp_risk_q_s - emp_risk_q_tilde_s_tilde)

    # Distances on INPUT IMAGES
    source_test_images, _ = get_all_data(source_test_loader)
    w1_dist_images = calculate_distance(source_test_images, adv_test_images, 'w1')
    mmd_dist_images = calculate_distance(source_test_images, adv_test_images, 'mmd')
    
    # True Risk Difference and extra metrics
    risk_q_p_tilde = calculate_risk(model_q, target_test_loader, device)
    acc_q_on_p_tilde = 1 - risk_q_p_tilde
    delta_r = np.abs(risk_q_p - risk_q_p_tilde)
    
    # Final Bounds
    bound_w1 = g_q + g_q_tilde + d_q_q_tilde + w1_dist_images
    tightness_w1 = bound_w1 / delta_r if delta_r > 1e-6 else np.inf
    bound_mmd = g_q + g_q_tilde + d_q_q_tilde + mmd_dist_images
    tightness_mmd = bound_mmd / delta_r if delta_r > 1e-6 else np.inf

    result = {
        'delta_r': delta_r, 'acc_q_on_p_tilde': acc_q_on_p_tilde,
        'g_q': g_q, 'g_q_tilde': g_q_tilde, 'd_q_q_tilde': d_q_q_tilde,
        'w1_dist_images': w1_dist_images, 'bound_w1': bound_w1, 'tightness_w1': tightness_w1,
        'mmd_dist_images': mmd_dist_images, 'bound_mmd': bound_mmd, 'tightness_mmd': tightness_mmd,
    }
    df = pd.DataFrame([result])
    # Save to the new experiment's directory
    os.makedirs('exp_11_adversarial_cifar/results', exist_ok=True)
    output_path = os.path.join('exp_11_adversarial_cifar/results', f'adversarial_cifar_eps_{args.attack_eps}.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\n--- CIFAR-10 Stress Test Complete ---")
    print(df.to_string())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run End-to-End Adversarial Stress Test on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs to train classifiers.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for data loaders.')
    parser.add_argument('--attack_eps', type=float, default=8/255, help='Epsilon for PGD attack (common value for CIFAR-10).')
    parser.add_argument('--attack_alpha', type=float, default=2/255, help='Alpha for PGD attack.')
    parser.add_argument('--attack_steps', type=int, default=10, help='Steps for PGD attack.')
    args = parser.parse_args()
    run_cifar_experiment(args)
