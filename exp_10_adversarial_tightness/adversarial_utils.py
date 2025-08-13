import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset

# --- Risk & Bound Calculation Utilities ---

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


def estimate_lipschitz_constant(model, data_loader, device):
    """
    Estimates the Lipschitz constant L_x of the model's feature extractor.
    """
    model.eval()
    max_norm = 0.0
    for data, _ in data_loader:
        data = data.to(device)
        data.requires_grad = True
        features = model(data)
        
        # We need a scalar output to compute the gradient, so we sum the features.
        # The choice of reduction doesn't matter as we are interested in the norm of the gradient.
        scalar_output = features.sum()
        
        gradients = grad(scalar_output, data, create_graph=False)[0]
        grad_norm = gradients.view(gradients.size(0), -1).norm(p=2, dim=1)
        
        current_max = grad_norm.max().item()
        if current_max > max_norm:
            max_norm = current_max
            
    return max_norm

def calculate_output_distance(model_q, model_q_tilde, features, device):
    """
    Calculates the L2 distance between the outputs of two models on the same features.
    """
    model_q.eval()
    model_q_tilde.eval()
    features = features.to(device)
    
    with torch.no_grad():
        outputs_q = model_q(features)
        outputs_q_tilde = model_q_tilde(features)
        
    return torch.norm(outputs_q - outputs_q_tilde, p=2, dim=1).mean().item()

def calculate_wasserstein1_distance(X, Y):
    """
    Computes the exact Wasserstein-1 distance between two empirical distributions.
    """
    # Cost matrix
    C = torch.cdist(X, Y, p=2)
    
    # linear_sum_assignment (Hungarian algorithm) finds the optimal transport plan
    row_ind, col_ind = linear_sum_assignment(C.cpu().numpy())
    
    # The optimal transport cost is the sum of costs for the optimal assignments
    w1_dist = C[row_ind, col_ind].sum() / X.size(0)
    
    return w1_dist.item()


def calculate_full_bound(g_q, g_q_tilde, l_x_q, l_x_q_tilde, w1_dist, output_dist):
    """
    Calculates the full theoretical bound from the methodology.
    """
    return g_q + g_q_tilde + (l_x_q + l_x_q_tilde) * w1_dist + output_dist

# --- Adversarial Shift Search ---

def find_worst_case_shift(model_q, model_q_tilde, feature_extractor, source_features, source_labels, 
                          data_loader, device, budget=100, shift_magnitude=0.1):
    """
    Searches for a feature shift that maximizes the bound-to-risk-change ratio.
    """
    worst_shift = None
    worst_ratio = -1.0
    
    l_x_q = estimate_lipschitz_constant(model_q, data_loader, device)
    l_x_q_tilde = estimate_lipschitz_constant(model_q_tilde, data_loader, device)

    # We need to calculate risk for the original data to get delta_r
    
    # Initial risks on the original (unshifted) test data
    risk_q_p = calculate_risk_linear(model_q, source_features, source_labels, device)
    
    for i in tqdm(range(budget), desc="Searching for worst-case shift"):
        # 1. Generate a candidate shift
        # We will test a simple shift: shifting a single feature dimension
        dim_to_shift = np.random.randint(0, source_features.shape[1])
        
        shifted_features = source_features.clone()
        shifted_features[:, dim_to_shift] += shift_magnitude * torch.randn(1).item()

        # 2. Compute true risk change (|Delta R|)
        risk_q_p_tilde = calculate_risk_linear(model_q, shifted_features, source_labels, device)
        delta_r = np.abs(risk_q_p - risk_q_p_tilde)

        if delta_r == 0:
            continue

        # 3. Compute the bound for this shift
        # Generalization gaps (can be pre-calculated, but for simplicity we recalculate)
        emp_risk_q_s = calculate_risk_linear(model_q, source_features, source_labels, device)
        g_q = np.abs(risk_q_p - emp_risk_q_s)

        # For g_q_tilde, we'd need to train a new model on shifted data. 
        # For simplicity in this search, we can approximate it or use a fixed value.
        # Here we'll assume it's similar to g_q for the search.
        g_q_tilde = g_q 

        w1_dist = calculate_wasserstein1_distance(source_features, shifted_features)
        
        output_dist = calculate_output_distance(model_q, model_q_tilde, shifted_features, device)

        bound = calculate_full_bound(g_q, g_q_tilde, l_x_q, l_x_q_tilde, w1_dist, output_dist)
        
        ratio = bound / delta_r

        # 4. Update worst-case if ratio is higher
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_shift = {
                'type': 'single_dim_shift',
                'dimension': dim_to_shift,
                'magnitude': shift_magnitude,
                'ratio': ratio,
                'bound': bound,
                'delta_r': delta_r,
                'w1_dist': w1_dist,
                'output_dist': output_dist
            }
            
    return worst_shift
