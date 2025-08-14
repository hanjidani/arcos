import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import os
import random
import torchattacks
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

# Import enhanced utilities
from adversarial_utils import (
    calculate_wasserstein1_distance_sinkhorn,
    calculate_output_distance_kl,
    estimate_lipschitz_constant_loss_based,
    calculate_expected_calibration_error,
    find_optimal_temperature,
    calculate_output_distance_kl_calibrated
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

# --- Standalone End-to-End Model ---
# No more separate feature extractor. This model is trained from scratch.
class EndToEndModel(nn.Module):
    def __init__(self, num_classes=10):
        super(EndToEndModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc_layers(x)
        return x
    
    def get_features(self, x):
        """Extract penultimate layer features for W1 computation."""
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc_layers[:-1](x)  # Exclude final classification layer
        return x

# --- Helper Functions ---

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

def calculate_expected_loss(model, loader, device, temperature=1.0):
    """Calculate expected loss for bound tightness analysis."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if temperature != 1.0:
                outputs = outputs / temperature
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
    
    return total_loss / total_samples if total_samples > 0 else 0.0

def get_all_data(loader):
    all_images = torch.cat([data for data, _ in loader], 0)
    all_labels = torch.cat([labels for _, labels in loader], 0)
    return all_images, all_labels

def calculate_distance(X, Y, metric='w1'):
    X_flat = X.view(X.size(0), -1)
    Y_flat = Y.view(Y.size(0), -1)
    
    if len(X) > 2000:
        indices = np.random.choice(len(X), 2000, replace=False)
        X_flat, Y_flat = X_flat[indices], Y_flat[indices]

    if metric == 'w1':
        # Use enhanced Sinkhorn Wasserstein for efficiency
        return calculate_wasserstein1_distance_sinkhorn(X_flat, Y_flat)
    elif metric == 'mmd':
        gamma = torch.median(torch.cdist(X_flat, X_flat)[X_flat.shape[0] > 1])
        K_XX = torch.exp(-torch.cdist(X_flat, X_flat)**2 / (2 * gamma**2))
        K_YY = torch.exp(-torch.cdist(Y_flat, Y_flat)**2 / (2 * gamma**2))
        K_XY = torch.exp(-torch.cdist(X_flat, Y_flat)**2 / (2 * gamma**2))
        mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
        return torch.sqrt(mmd2).item() if mmd2 > 0 else 0.0

def generate_adversarial_data(model, loader, device, eps, alpha, steps, attack_type='pgd'):
    """Generate adversarial data with proper model.eval() handling."""
    model.eval()  # Ensure model is in eval mode for adversarial generation
    
    if attack_type == 'pgd':
        attack = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)
    elif attack_type == 'fgsm':
        attack = torchattacks.FGSM(model, eps=eps)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
    
    adv_images = []
    adv_labels = []
    
    for images, labels in tqdm(loader, desc=f"Generating {attack_type.upper()} attacks (eps={eps})"):
        images, labels = images.to(device), labels.to(device)
        adv_batch = attack(images, labels)
        adv_images.append(adv_batch.cpu())
        adv_labels.append(labels.cpu())
    
    return torch.cat(adv_images), torch.cat(adv_labels)

def run_autoattack_evaluation(model, loader, device, eps):
    """Run AutoAttack for comprehensive adversarial evaluation."""
    try:
        from autoattack import AutoAttack
        
        # Ensure model is in eval mode before crafting adversarial examples
        model.eval()
        
        # Prepare data
        images, labels = get_all_data(loader)
        images = images.to(device)
        labels = labels.to(device)
        
        # Run AutoAttack
        adversary = AutoAttack(model, norm='Linf', eps=eps, version='standard')
        x_adv = adversary.run_standard_evaluation(images, labels, bs=128)
        
        # Calculate robust accuracy
        with torch.no_grad():
            outputs = model(x_adv)
            _, predicted = torch.max(outputs.data, 1)
            robust_acc = (predicted == labels).sum().item() / labels.size(0)
        
        return 1 - robust_acc  # Return robust risk
    except ImportError:
        print("AutoAttack not available, skipping...")
        return None
    except Exception as e:
        print(f"AutoAttack evaluation failed: {e}")
        return None

# --- Main Experiment ---

def run_end_to_end_experiment(args):
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

    # 1. Load Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    source_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    source_test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # 2. Train Source Model (Q) from Scratch
    model_q = train_model(EndToEndModel(), source_train_loader, device, args.epochs, "Training source model Q")

    # 3. Generate Adversarial Dataset (P_tilde) with multiple epsilon values
    print("\nCreating Adversarial Distribution P_tilde...")
    
    # Test multiple epsilon values for comprehensive evaluation
    eps_values = [0.1, 2/255, 4/255, 8/255, 12/255] if args.attack_eps == 0.1 else [args.attack_eps]
    
    all_results = []
    
    for eps in eps_values:
        print(f"\n--- Testing epsilon = {eps:.4f} ---")
        
        # Generate adversarial data
        adv_train_images, adv_train_labels = generate_adversarial_data(
            model_q, source_train_loader, device, eps, args.attack_alpha, args.attack_steps
        )
        adv_test_images, adv_test_labels = generate_adversarial_data(
            model_q, source_test_loader, device, eps, args.attack_alpha, args.attack_steps
        )

        target_train_loader = DataLoader(TensorDataset(adv_train_images, adv_train_labels), 
                                       batch_size=args.batch_size, shuffle=True)
        target_test_loader = DataLoader(TensorDataset(adv_test_images, adv_test_labels), 
                                      batch_size=args.batch_size)

        # 4. Train Target Model (Q_tilde) from Scratch on Adversarial Data
        model_q_tilde = train_model(EndToEndModel(), target_train_loader, device, args.epochs, 
                                  f"Training target model Q_tilde (eps={eps})")

        # 5. Calculate All Metrics for the Bound
        print(f"\nCalculating Final Metrics for eps={eps}...")
        
        # Risks and Gaps
        risk_q_p = calculate_risk(model_q, source_test_loader, device)
        emp_risk_q_s = calculate_risk(model_q, source_train_loader, device)
        g_q = np.abs(risk_q_p - emp_risk_q_s)

        risk_q_tilde_p_tilde = calculate_risk(model_q_tilde, target_test_loader, device)
        emp_risk_q_tilde_s_tilde = calculate_risk(model_q_tilde, target_train_loader, device)
        g_q_tilde = np.abs(risk_q_tilde_p_tilde - emp_risk_q_tilde_s_tilde)
        
        d_q_q_tilde = np.abs(emp_risk_q_s - emp_risk_q_tilde_s_tilde)

        # Expected losses for bound tightness
        expected_loss_q = calculate_expected_loss(model_q, source_test_loader, device)
        expected_loss_q_tilde = calculate_expected_loss(model_q_tilde, target_test_loader, device)
        
        # Distances on INPUT IMAGES
        source_test_images, _ = get_all_data(source_test_loader)
        w1_dist_images = calculate_distance(source_test_images, adv_test_images, 'w1')
        mmd_dist_images = calculate_distance(source_test_images, adv_test_images, 'mmd')
        
        # True Risk Differences - TWO different gaps for proper tightness analysis
        # 1. Distribution shift gap: same model across different distributions
        risk_q_p_tilde = calculate_risk(model_q, target_test_loader, device)
        acc_q_on_p_tilde = 1 - risk_q_p_tilde
        delta_r_shift = np.abs(risk_q_p - risk_q_p_tilde)
        
        # 2. Model comparison gap: different models on same distribution (what the bound controls)
        risk_q_tilde_p = calculate_risk(model_q_tilde, source_test_loader, device)
        delta_r_model = np.abs(risk_q_p - risk_q_tilde_p)
        
        # Lipschitz constants for bound analysis
        l_x_q = estimate_lipschitz_constant_loss_based(model_q, source_test_loader, device)
        l_x_q_tilde = estimate_lipschitz_constant_loss_based(model_q_tilde, target_test_loader, device)
        
        # Output distance using KL divergence (more theoretically sound)
        output_dist_kl = calculate_output_distance_kl(model_q, model_q_tilde, adv_test_images, device)
        
        # Temperature calibration and ECE
        ece_q, ace_q = calculate_expected_calibration_error(model_q, source_test_loader, device)
        ece_q_tilde, ace_q_tilde = calculate_expected_calibration_error(model_q_tilde, target_test_loader, device)
        
        # Find optimal temperature for KL divergence
        best_temp, best_ece, temp_results = find_optimal_temperature(model_q, source_test_loader, device)
        
        # Recompute KL divergence with optimal temperature
        output_dist_kl_opt, temp_used, ece_q_opt, ece_q_tilde_opt = calculate_output_distance_kl_calibrated(
            model_q, model_q_tilde, adv_test_images, device, 
            temperature=best_temp, val_loader=source_test_loader, auto_calibrate=False
        )
        
        # Final Bounds (four-term decomposition)
        bound_w1 = g_q + g_q_tilde + (l_x_q + l_x_q_tilde) * w1_dist_images + output_dist_kl_opt
        bound_mmd = g_q + g_q_tilde + (l_x_q + l_x_q_tilde) * mmd_dist_images + output_dist_kl_opt
        
        # TWO tightness metrics for proper analysis
        # 1. Model comparison tightness (what the bound controls)
        tightness_model_w1 = bound_w1 / delta_r_model if delta_r_model > 1e-6 else np.inf
        tightness_model_mmd = bound_mmd / delta_r_model if delta_r_model > 1e-6 else np.inf
        
        # 2. Distribution shift tightness (Lipschitz * W1 term only)
        lipschitz_w1_term = (l_x_q + l_x_q_tilde) * w1_dist_images
        lipschitz_mmd_term = (l_x_q + l_x_q_tilde) * mmd_dist_images
        tightness_shift_w1 = lipschitz_w1_term / delta_r_shift if delta_r_shift > 1e-6 else np.inf
        tightness_shift_mmd = lipschitz_mmd_term / delta_r_shift if delta_r_shift > 1e-6 else np.inf
        
        # AutoAttack evaluation
        autoattack_risk = run_autoattack_evaluation(model_q, source_test_loader, device, eps)

        # Per-bucket tightness analysis
        try:
            from adversarial_utils import analyze_bound_tightness_by_buckets
            
            bucket_analysis = analyze_bound_tightness_by_buckets(
                model_q, model_q_tilde, source_test_images, adv_test_images,
                source_test_labels, device, g_q, g_q_tilde, l_x_q, l_x_q_tilde
            )
            
            # Add bucket summary to results
            bucket_summary = {}
            for bucket_name, bucket_info in bucket_analysis.items():
                if 'error' not in bucket_info:
                    bucket_summary[f'{bucket_name}_tightness'] = bucket_info.get('tightness', np.inf)
                    bucket_summary[f'{bucket_name}_size'] = bucket_info.get('size', 0)
                    bucket_summary[f'{bucket_name}_avg_confidence'] = bucket_info.get('avg_confidence', 0.0)
                    bucket_summary[f'{bucket_name}_avg_entropy'] = bucket_info.get('avg_entropy', 0.0)
            
            result.update(bucket_summary)
            
        except Exception as e:
            print(f"Warning: Bucket analysis failed: {e}")
            bucket_analysis = None

        result = {
            'epsilon': eps,
            # Risk differences for proper tightness analysis
            'delta_r_shift': delta_r_shift,  # Same model, different distributions
            'delta_r_model': delta_r_model,  # Different models, same distribution (bound target)
            'acc_q_on_p_tilde': acc_q_on_p_tilde,
            # Bound components (four-term decomposition)
            'g_q': g_q, 
            'g_q_tilde': g_q_tilde, 
            'd_q_q_tilde': d_q_q_tilde,
            'w1_dist_images': w1_dist_images, 
            'bound_w1': bound_w1, 
            'bound_mmd': bound_mmd,
            # TWO tightness metrics for proper analysis
            'tightness_model_w1': tightness_model_w1,  # Bound tightness (what we control)
            'tightness_model_mmd': tightness_model_mmd,
            'tightness_shift_w1': tightness_shift_w1,  # Distribution shift tightness
            'tightness_shift_mmd': tightness_shift_mmd,
            # Lipschitz terms for analysis
            'lipschitz_w1_term': lipschitz_w1_term,
            'lipschitz_mmd_term': lipschitz_mmd_term,
            # Expected losses for bound analysis
            'expected_loss_q': expected_loss_q,
            'expected_loss_q_tilde': expected_loss_q_tilde,
            'lipschitz_q': l_x_q,
            'lipschitz_q_tilde': l_x_q_tilde,
            # Output distance with temperature calibration
            'output_dist_kl': output_dist_kl,
            'output_dist_kl_optimal': output_dist_kl_opt,
            'temperature_used': temp_used,
            # Calibration metrics
            'ece_q': ece_q,
            'ace_q': ace_q,
            'ece_q_tilde': ece_q_tilde,
            'ace_q_tilde': ace_q_tilde,
            'ece_q_optimal': ece_q_opt,
            'ece_q_tilde_optimal': ece_q_tilde_opt,
            # Attack evaluation
            'autoattack_risk': autoattack_risk,
            # Space information for clarity
            'w1_space': 'images',  # W1 computed on image space
            'lipschitz_space': 'images',  # Lipschitz wrt image space
            # Sinkhorn parameters for reproducibility
            'sinkhorn_epsilon': 0.01,
            'sinkhorn_max_iter': 100,
            'sinkhorn_tol': 1e-6
        }
        
        all_results.append(result)
        
        print(f"Results for eps={eps}:")
        print(f"  Distribution Shift Gap (same model): {delta_r_shift:.4f}")
        print(f"  Model Comparison Gap (bound target): {delta_r_model:.4f}")
        print(f"  W1 Bound: {bound_w1:.4f}")
        print(f"  Model Tightness (bound control): {tightness_model_w1:.4f}")
        print(f"  Shift Tightness (Lipschitz*W1): {tightness_shift_w1:.4f}")
        if autoattack_risk is not None:
            print(f"  AutoAttack Risk: {autoattack_risk:.4f}")

    # 6. Save Comprehensive Results
    df = pd.DataFrame(all_results)
    os.makedirs('results', exist_ok=True)
    output_path = os.path.join('results', f'adversarial_end_to_end_comprehensive.csv')
    df.to_csv(output_path, index=False)
    
    print("\n--- End-to-End Stress Test Complete ---")
    print(f"Results saved to {output_path}")
    print("\nSummary:")
    print(df[['epsilon', 'delta_r_model', 'delta_r_shift', 'tightness_model_w1', 'tightness_shift_w1']].to_string())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run End-to-End Adversarial Stress Test')
    parser.add_argument('--epochs', type=int, default=5, help='Epochs to train classifiers.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for data loaders.')
    parser.add_argument('--attack_eps', type=float, default=0.1, help='Epsilon for PGD attack.')
    parser.add_argument('--attack_alpha', type=float, default=0.01, help='Alpha for PGD attack.')
    parser.add_argument('--attack_steps', type=int, default=10, help='Steps for PGD attack.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()
    run_end_to_end_experiment(args)
