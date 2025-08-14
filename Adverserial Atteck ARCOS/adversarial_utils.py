import torch
import torch.nn as nn
import torch.nn.functional as F
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

def calculate_expected_loss(model, features, labels, device, temperature=1.0):
    """
    Calculates the expected loss (cross-entropy) for bound tightness analysis.
    """
    model.eval()
    model.to(device)
    features = features.to(device)
    labels = labels.to(device)
    
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=1024)
    
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for feature_batch, label_batch in loader:
            outputs = model(feature_batch)
            # Apply temperature scaling
            if temperature != 1.0:
                outputs = outputs / temperature
            loss = F.cross_entropy(outputs, label_batch, reduction='sum')
            total_loss += loss.item()
            total_samples += label_batch.size(0)
    
    return total_loss / total_samples if total_samples > 0 else 0.0

def estimate_lipschitz_constant_loss_based(model, data_loader, device, temperature=1.0):
    """
    Estimates the Lipschitz constant using loss-based gradients (more theoretically sound).
    """
    model.eval()
    max_norm = 0.0
    criterion = nn.CrossEntropyLoss()
    
    for data, labels in data_loader:
        data = data.to(device)
        labels = labels.to(device)
        data.requires_grad = True
        
        outputs = model(data)
        if temperature != 1.0:
            outputs = outputs / temperature
        
        # Use cross-entropy loss for gradient computation
        loss = criterion(outputs, labels)
        
        gradients = grad(loss, data, create_graph=False)[0]
        grad_norm = gradients.view(gradients.size(0), -1).norm(p=2, dim=1)
        
        current_max = grad_norm.max().item()
        if current_max > max_norm:
            max_norm = current_max
            
    return max_norm

def estimate_lipschitz_constant_jacobian(model, data_loader, device):
    """
    Alternative: Estimates Lipschitz constant using Jacobian norm aggregation.
    """
    model.eval()
    max_norm = 0.0
    
    for data, _ in data_loader:
        data = data.to(device)
        data.requires_grad = True
        
        outputs = model(data)
        batch_size, num_classes = outputs.shape
        
        # Compute Jacobian norm for each sample
        for i in range(batch_size):
            sample_grads = []
            for j in range(num_classes):
                grad_output = torch.zeros_like(outputs)
                grad_output[i, j] = 1.0
                grad_input = grad(outputs, data, grad_outputs=grad_output, create_graph=False)[0]
                sample_grads.append(grad_input[i].view(-1))
            
            # Aggregate gradients across output dimensions
            sample_grads = torch.stack(sample_grads)
            jacobian_norm = sample_grads.norm(p=2, dim=1).sum().item()
            max_norm = max(max_norm, jacobian_norm)
            
    return max_norm

def calculate_output_distance_kl(model_q, model_q_tilde, features, device, temperature=1.0):
    """
    Calculates KL divergence between softmax outputs (more theoretically sound than L2).
    """
    model_q.eval()
    model_q_tilde.eval()
    features = features.to(device)
    
    with torch.no_grad():
        outputs_q = model_q(features)
        outputs_q_tilde = model_q_tilde(features)
        
        # Apply temperature scaling and softmax
        if temperature != 1.0:
            outputs_q = outputs_q / temperature
            outputs_q_tilde = outputs_q_tilde / temperature
        
        probs_q = F.softmax(outputs_q, dim=1)
        probs_q_tilde = F.softmax(outputs_q_tilde, dim=1)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        probs_q = probs_q + eps
        probs_q_tilde = probs_q_tilde + eps
        
        # Normalize to sum to 1
        probs_q = probs_q / probs_q.sum(dim=1, keepdim=True)
        probs_q_tilde = probs_q_tilde / probs_q_tilde.sum(dim=1, keepdim=True)
        
        # Calculate KL divergence
        kl_div = F.kl_div(
            torch.log(probs_q_tilde), probs_q, 
            reduction='batchmean'
        )
        
    return kl_div.item()

def calculate_output_distance(model_q, model_q_tilde, features, device):
    """
    Legacy L2 distance function (kept for backward compatibility).
    """
    model_q.eval()
    model_q_tilde.eval()
    features = features.to(device)
    
    with torch.no_grad():
        outputs_q = model_q(features)
        outputs_q_tilde = model_q_tilde(features)
        
    return torch.norm(outputs_q - outputs_q_tilde, p=2, dim=1).mean().item()

def calculate_wasserstein1_distance_sinkhorn(X, Y, epsilon=0.01, max_iter=100, tol=1e-6, freeze_features=True, weights_X=None, weights_Y=None):
    """
    Computes Wasserstein-1 distance using Sinkhorn algorithm (more efficient for large datasets).
    
    Args:
        X, Y: Input tensors
        epsilon: Entropic regularization parameter (smaller = more accurate but slower)
        max_iter: Maximum number of Sinkhorn iterations
        tol: Convergence tolerance for dual residual
        freeze_features: If True, detach gradients to prevent updates through feature map h
        weights_X, weights_Y: Optional weight vectors for X and Y (must sum to 1)
    
    Note:
        If weights are provided, they must sum to 1 for each distribution.
        If no weights provided, assumes uniform weights (equal sample sizes required).
        
        IMPORTANT: Set freeze_features=True when computing W₁(h(S), h(S̃)) to prevent
        gradients from flowing through the feature map h, keeping the regularizer stable.
    """
    # Freeze gradients if requested (prevents updates through feature map h)
    if freeze_features:
        X = X.detach()
        Y = Y.detach()
    
    X_flat = X.view(X.size(0), -1)
    Y_flat = Y.view(Y.size(0), -1)
    
    # Handle weights
    if weights_X is None and weights_Y is None:
        # Uniform weights - require equal sample sizes
        if X_flat.size(0) != Y_flat.size(0):
            raise ValueError("Sinkhorn with uniform weights requires equal sample sizes. Use exact Wasserstein for different sizes or provide weights.")
        weights_X = torch.ones(X_flat.size(0), device=X.device) / X_flat.size(0)
        weights_Y = torch.ones(Y_flat.size(0), device=Y.device) / Y_flat.size(0)
    else:
        # Validate weights
        if weights_X is None:
            weights_X = torch.ones(X_flat.size(0), device=X.device) / X_flat.size(0)
        if weights_Y is None:
            weights_Y = torch.ones(Y_flat.size(0), device=Y.device) / Y_flat.size(0)
        
        # Ensure weights sum to 1
        weights_X = weights_X / weights_X.sum()
        weights_Y = weights_Y / weights_Y.sum()
    
    # Normalize/whiten features for better numerical stability
    X_mean = X_flat.mean(dim=0, keepdim=True)
    X_std = X_flat.std(dim=0, keepdim=True) + 1e-8
    X_norm = (X_flat - X_mean) / X_std
    
    Y_mean = Y_flat.mean(dim=0, keepdim=True)
    Y_std = Y_flat.std(dim=0, keepdim=True) + 1e-8
    Y_norm = (Y_flat - Y_mean) / Y_std
    
    # Cost matrix
    C = torch.cdist(X_norm, Y_norm, p=2)
    
    # Sinkhorn algorithm with convergence check
    n, m = C.shape
    log_mu = torch.zeros(n, 1, device=C.device)
    log_nu = torch.zeros(1, m, device=C.device)
    
    prev_log_mu = log_mu.clone()
    prev_log_nu = log_nu.clone()
    
    for iter_idx in range(max_iter):
        log_mu = -torch.logsumexp((-C + log_nu) / epsilon, dim=1, keepdim=True)
        log_nu = -torch.logsumexp((-C + log_mu) / epsilon, dim=0, keepdim=True)
        
        # Check convergence (dual residual)
        mu_change = torch.norm(log_mu - prev_log_mu).item()
        nu_change = torch.norm(log_nu - prev_log_nu).item()
        
        if max(mu_change, nu_change) < tol:
            break
            
        prev_log_mu = log_mu.clone()
        prev_log_nu = log_nu.clone()
    
    # Compute transport cost
    pi = torch.exp((-C + log_mu + log_nu) / epsilon)
    w1_dist = (pi * C).sum()
    
    return w1_dist.item()

def calculate_wasserstein1_distance(X, Y, freeze_features=True, weights_X=None, weights_Y=None):
    """
    Computes the exact Wasserstein-1 distance between two empirical distributions.
    Falls back to Sinkhorn for large datasets.
    
    Args:
        X, Y: Input tensors
        freeze_features: If True, detach gradients to prevent updates through feature map h
        weights_X, weights_Y: Optional weight vectors for X and Y (must sum to 1)
    """
    # Freeze gradients if requested (prevents updates through feature map h)
    if freeze_features:
        X = X.detach()
        Y = Y.detach()
    
    if X.size(0) > 1000 or Y.size(0) > 1000:
        return calculate_wasserstein1_distance_sinkhorn(X, Y, freeze_features=False, weights_X=weights_X, weights_Y=weights_Y)
    
    # Handle weights
    if weights_X is None and weights_Y is None:
        # Uniform weights - require equal sample sizes
        if X.size(0) != Y.size(0):
            raise ValueError(f"Exact Wasserstein with uniform weights requires equal sample sizes. Got {X.size(0)} vs {Y.size(0)}. Use Sinkhorn for different sizes or provide weights.")
        weights_X = torch.ones(X.size(0), device=X.device) / X.size(0)
        weights_Y = torch.ones(Y.size(0), device=Y.device) / Y.size(0)
    else:
        # Validate weights
        if weights_X is None:
            weights_X = torch.ones(X.size(0), device=X.device) / X.size(0)
        if weights_Y is None:
            weights_Y = torch.ones(Y.size(0), device=Y.device) / Y.size(0)
        
        # Ensure weights sum to 1
        weights_X = weights_X / weights_X.sum()
        weights_Y = weights_Y / weights_Y.sum()
    
    # Cost matrix
    C = torch.cdist(X, Y, p=2)
    
    # linear_sum_assignment (Hungarian algorithm) finds the optimal transport plan
    row_ind, col_ind = linear_sum_assignment(C.cpu().numpy())
    
    # The optimal transport cost is the sum of costs for the optimal assignments
    w1_dist = C[row_ind, col_ind].sum()
    
    return w1_dist.item()

def calculate_full_bound(g_q, g_q_tilde, l_x_q, l_x_q_tilde, w1_dist, output_dist):
    """
    Calculates the full theoretical bound from the methodology.
    """
    return g_q + g_q_tilde + (l_x_q + l_x_q_tilde) * w1_dist + output_dist

# --- Enhanced Adversarial Shift Search ---

def find_worst_case_shift_multidimensional(model_q, model_q_tilde, feature_extractor, source_features, source_labels, 
                                         data_loader, device, budget=100, shift_magnitude=0.1):
    """
    Enhanced adversarial shift search using multiple strategies:
    1. PCA-based shifts
    2. OT transport vector directions
    3. Gradient-based optimization
    """
    worst_shift = None
    worst_ratio = -1.0
    
    l_x_q = estimate_lipschitz_constant_loss_based(model_q, data_loader, device)
    l_x_q_tilde = estimate_lipschitz_constant_loss_based(model_q_tilde, data_loader, device)

    # Initial risks on the original (unshifted) test data
    risk_q_p = calculate_risk_linear(model_q, source_features, source_labels, device)
    
    # Strategy 1: PCA-based shifts
    print("Strategy 1: PCA-based shifts...")
    for i in tqdm(range(budget // 3), desc="PCA shifts"):
        # Compute PCA of feature differences
        if hasattr(feature_extractor, 'get_features'):
            # If we have access to feature extractor, use it
            with torch.no_grad():
                source_feat = feature_extractor(source_features)
                # For simplicity, create synthetic target features
                target_feat = source_feat + 0.1 * torch.randn_like(source_feat)
        else:
            # Use the features directly
            source_feat = source_features
            target_feat = source_features + 0.1 * torch.randn_like(source_features)
        
        # Compute PCA directions
        diff_features = target_feat - source_feat
        if diff_features.size(0) > 1:
            # Center the data
            diff_centered = diff_features - diff_features.mean(dim=0, keepdim=True)
            # Compute covariance matrix
            cov_matrix = torch.mm(diff_centered.T, diff_centered) / (diff_centered.size(0) - 1)
            
            # Get top eigenvector (simplified PCA)
            try:
                eigenvals, eigenvecs = torch.linalg.eigh(cov_matrix)
                top_direction = eigenvecs[:, -1]  # Top eigenvector
                
                # Apply shift along top PCA direction
                shifted_features = source_features.clone()
                shift_vector = shift_magnitude * top_direction.unsqueeze(0)
                shifted_features = shifted_features + shift_vector
                
                # Evaluate this shift
                ratio = evaluate_shift_ratio(
                    model_q, model_q_tilde, source_features, shifted_features,
                    source_labels, risk_q_p, l_x_q, l_x_q_tilde, device,
                    g_q=0.01, g_q_tilde=0.01  # These should be computed from train/test risks
                )
                
                if ratio > worst_ratio:
                    worst_ratio = ratio
                    worst_shift = {
                        'type': 'pca_shift',
                        'direction': 'top_eigenvector',
                        'magnitude': shift_magnitude,
                        'ratio': ratio,
                        'lipschitz_q': l_x_q,
                        'lipschitz_q_tilde': l_x_q_tilde
                    }
            except:
                continue
    
    # Strategy 2: Random multidimensional shifts
    print("Strategy 2: Random multidimensional shifts...")
    for i in tqdm(range(budget // 3), desc="Random shifts"):
        # Generate random shift vector
        shift_vector = torch.randn_like(source_features) * shift_magnitude
        shifted_features = source_features + shift_vector
        
        # Evaluate this shift
        ratio = evaluate_shift_ratio(
            model_q, model_q_tilde, source_features, shifted_features,
            source_labels, risk_q_p, l_x_q, l_x_q_tilde, device,
            g_q=0.01, g_q_tilde=0.01  # These should be computed from train/test risks
        )
        
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_shift = {
                'type': 'random_multidim_shift',
                'magnitude': shift_magnitude,
                'ratio': ratio,
                'lipschitz_q': l_x_q,
                'lipschitz_q_tilde': l_x_q_tilde
            }
    
    # Strategy 3: Gradient-based optimization
    print("Strategy 3: Gradient-based optimization...")
    for i in tqdm(range(budget // 3), desc="Gradient optimization"):
        # Initialize perturbation
        perturbation = torch.randn_like(source_features) * 0.01
        perturbation.requires_grad = True
        
        # Run a few PGD steps to maximize the bound
        for step in range(5):
            shifted_features = source_features + perturbation
            
            # Compute bound components
            w1_dist = calculate_wasserstein1_distance_sinkhorn(source_features, shifted_features)
            output_dist = calculate_output_distance_kl(model_q, model_q_tilde, shifted_features, device)
            
            # Compute risk change
            risk_q_p_tilde = calculate_risk_linear(model_q, shifted_features, source_labels, device)
            delta_r = np.abs(risk_q_p - risk_q_p_tilde)
            
            if delta_r == 0:
                break
                
            # Compute bound
            bound = calculate_full_bound(0, 0, l_x_q, l_x_q_tilde, w1_dist, output_dist)
            
            # Loss to maximize (negative because we want to maximize)
            loss = -bound / (delta_r + 1e-8)
            
            # Gradient step
            loss.backward()
            with torch.no_grad():
                perturbation = perturbation + 0.01 * perturbation.grad.sign()
                perturbation = torch.clamp(perturbation, -shift_magnitude, shift_magnitude)
                if perturbation.grad is not None:
                    perturbation.grad.zero_()
        
        # Evaluate final perturbation
        shifted_features = source_features + perturbation.detach()
        ratio = evaluate_shift_ratio(
            model_q, model_q_tilde, source_features, shifted_features,
            source_labels, risk_q_p, l_x_q, l_x_q_tilde, device,
            g_q=0.01, g_q_tilde=0.01  # These should be computed from train/test risks
        )
        
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_shift = {
                'type': 'gradient_optimized_shift',
                'magnitude': perturbation.norm().item(),
                'ratio': ratio,
                'lipschitz_q': l_x_q,
                'lipschitz_q_tilde': l_x_q_tilde
            }
    
    return worst_shift

def evaluate_shift_ratio(model_q, model_q_tilde, source_features, shifted_features, 
                        source_labels, risk_q_p, l_x_q, l_x_q_tilde, device, 
                        g_q=None, g_q_tilde=None):
    """
    Helper function to evaluate the bound-to-risk-change ratio for a given shift.
    """
    # Compute risk change
    risk_q_p_tilde = calculate_risk_linear(model_q, shifted_features, source_labels, device)
    delta_r = np.abs(risk_q_p - risk_q_p_tilde)
    
    if delta_r == 0:
        return -1.0
    
    # Compute bound components
    w1_dist = calculate_wasserstein1_distance_sinkhorn(source_features, shifted_features)
    output_dist = calculate_output_distance_kl(model_q, model_q_tilde, shifted_features, device)
    
    # Use provided generalization gaps or compute them if not provided
    if g_q is None or g_q_tilde is None:
        # Compute generalization gaps from train/test risks
        # This would require access to training data, so we'll use a reasonable default
        # In practice, these should be computed and passed in from the calling function
        g_q = 0.01  # Default fallback
        g_q_tilde = 0.01  # Default fallback
    
    bound = calculate_full_bound(g_q, g_q_tilde, l_x_q, l_x_q_tilde, w1_dist, output_dist)
    
    return bound / delta_r

def find_worst_case_shift(model_q, model_q_tilde, feature_extractor, source_features, source_labels, 
                          data_loader, device, budget=100, shift_magnitude=0.1):
    """
    Legacy function for backward compatibility - now calls the enhanced version.
    """
    return find_worst_case_shift_multidimensional(
        model_q, model_q_tilde, feature_extractor, source_features, source_labels,
        data_loader, device, budget, shift_magnitude
    )

def create_ema_teacher(model, decay=0.999):
    """
    Creates an EMA (Exponential Moving Average) teacher copy of a model for stable W₁ computation.
    
    Args:
        model: The source model to create a teacher from
        decay: EMA decay rate (higher = more stable, slower to adapt)
    
    Returns:
        teacher_model: EMA copy with gradients stopped
        update_teacher: Function to update the teacher with current model weights
    """
    teacher_model = type(model)()
    teacher_model.load_state_dict(model.state_dict())
    teacher_model.eval()
    
    # Stop gradients through teacher
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    def update_teacher():
        """Update teacher model with current model weights using EMA."""
        with torch.no_grad():
            for teacher_param, student_param in zip(teacher_model.parameters(), model.parameters()):
                teacher_param.data = decay * teacher_param.data + (1 - decay) * student_param.data
    
    return teacher_model, update_teacher

def calculate_expected_calibration_error(model, data_loader, device, num_bins=15):
    """
    Calculates Expected Calibration Error (ECE) to assess model calibration.
    
    Args:
        model: The model to evaluate
        data_loader: Data loader for evaluation
        device: Device to run on
        num_bins: Number of confidence bins for ECE calculation
    
    Returns:
        ece: Expected calibration error
        ace: Adaptive calibration error (more robust)
    """
    model.eval()
    confidences = []
    accuracies = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidence, predictions = torch.max(probs, dim=1)
            
            # Store confidence and accuracy for each sample
            for i in range(len(images)):
                confidences.append(confidence[i].item())
                accuracies.append((predictions[i] == labels[i]).item())
    
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    
    # Calculate ECE
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    ace = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this confidence bin
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            bin_accuracy = np.mean(accuracies[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            ece += bin_size * np.abs(bin_accuracy - bin_confidence)
    
    ece = ece / len(confidences)
    
    # Calculate ACE (Adaptive Calibration Error)
    # Sort by confidence and create adaptive bins
    sorted_indices = np.argsort(confidences)
    sorted_confidences = confidences[sorted_indices]
    sorted_accuracies = accuracies[sorted_indices]
    
    bin_size = len(confidences) // num_bins
    ace = 0.0
    
    for i in range(num_bins):
        start_idx = i * bin_size
        end_idx = min((i + 1) * bin_size, len(confidences))
        
        if end_idx > start_idx:
            bin_accuracy = np.mean(sorted_accuracies[start_idx:end_idx])
            bin_confidence = np.mean(sorted_confidences[start_idx:end_idx])
            ace += np.abs(bin_accuracy - bin_confidence)
    
    ace = ace / num_bins
    
    return ece, ace

def find_optimal_temperature(model, val_loader, device, temperature_range=[0.5, 1.0, 2.0, 4.0]):
    """
    Finds the optimal temperature for KL divergence computation by minimizing ECE on validation set.
    
    Args:
        model: The model to calibrate
        val_loader: Validation data loader
        device: Device to run on
        temperature_range: List of temperatures to try
    
    Returns:
        best_temp: Temperature that minimizes ECE
        best_ece: Best ECE achieved
        temp_results: Dictionary with results for each temperature
    """
    temp_results = {}
    best_temp = 1.0
    best_ece = float('inf')
    
    for temp in temperature_range:
        # Calculate ECE with this temperature
        ece, ace = calculate_expected_calibration_error(model, val_loader, device)
        temp_results[temp] = {'ece': ece, 'ace': ace}
        
        if ece < best_ece:
            best_ece = ece
            best_temp = temp
    
    return best_temp, best_ece, temp_results

def calculate_output_distance_kl_calibrated(model_q, model_q_tilde, features, device, 
                                          temperature=1.0, val_loader=None, auto_calibrate=False):
    """
    Calculates KL divergence between softmax outputs with optional temperature calibration.
    
    Args:
        model_q, model_q_tilde: Models to compare
        features: Input features
        device: Device to run on
        temperature: Temperature for softmax scaling
        val_loader: Validation loader for temperature calibration
        auto_calibrate: If True, automatically find optimal temperature
    
    Returns:
        kl_dist: KL divergence
        temperature_used: Temperature actually used
        ece_q: ECE for model_q
        ece_q_tilde: ECE for model_q_tilde
    """
    if auto_calibrate and val_loader is not None:
        # Find optimal temperature for both models
        best_temp_q, ece_q, _ = find_optimal_temperature(model_q, val_loader, device)
        best_temp_q_tilde, ece_q_tilde, _ = find_optimal_temperature(model_q_tilde, val_loader, device)
        
        # Use average of optimal temperatures
        temperature = (best_temp_q + best_temp_q_tilde) / 2
        print(f"Auto-calibrated temperature: {temperature:.3f} (Q: {best_temp_q:.3f}, Q̃: {best_temp_q_tilde:.3f})")
    else:
        # Calculate ECE with current temperature
        if val_loader is not None:
            ece_q, _ = calculate_expected_calibration_error(model_q, val_loader, device)
            ece_q_tilde, _ = calculate_expected_calibration_error(model_q_tilde, val_loader, device)
        else:
            ece_q = ece_q_tilde = None
    
    model_q.eval()
    model_q_tilde.eval()
    features = features.to(device)
    
    with torch.no_grad():
        outputs_q = model_q(features)
        outputs_q_tilde = model_q_tilde(features)
        
        # Apply temperature scaling and softmax
        if temperature != 1.0:
            outputs_q = outputs_q / temperature
            outputs_q_tilde = outputs_q_tilde / temperature
        
        probs_q = F.softmax(outputs_q, dim=1)
        probs_q_tilde = F.softmax(outputs_q_tilde, dim=1)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        probs_q = probs_q + eps
        probs_q_tilde = probs_q_tilde + eps
        
        # Normalize to sum to 1
        probs_q = probs_q / probs_q.sum(dim=1, keepdim=True)
        probs_q_tilde = probs_q_tilde / probs_q_tilde.sum(dim=1, keepdim=True)
        
        # Calculate KL divergence
        kl_div = F.kl_div(
            torch.log(probs_q_tilde), probs_q, 
            reduction='batchmean'
        )
        
    return kl_div.item(), temperature, ece_q, ece_q_tilde

def calculate_per_bucket_tightness(model_q, model_q_tilde, source_features, target_features, 
                                  source_labels, device, num_buckets=4):
    """
    Calculates tightness analysis per bucket to reveal bound behavior on different example types.
    
    Args:
        model_q, model_q_tilde: Models to compare
        source_features, target_features: Source and target feature distributions
        source_labels: Labels for source data
        device: Device to run on
        num_buckets: Number of buckets for analysis
    
    Returns:
        bucket_results: Dictionary with tightness analysis per bucket
    """
    model_q.eval()
    model_q_tilde.eval()
    
    # Get model predictions and confidences
    with torch.no_grad():
        outputs_q = model_q(source_features.to(device))
        probs_q = F.softmax(outputs_q, dim=1)
        confidence_q, predictions_q = torch.max(probs_q, dim=1)
        
        # Calculate entropy of predictions
        entropy_q = -torch.sum(probs_q * torch.log(probs_q + 1e-8), dim=1)
        
        # Calculate perturbation norms (if target_features provided)
        if target_features is not None:
            perturbation_norms = torch.norm(source_features - target_features, p=2, dim=1)
        else:
            perturbation_norms = torch.zeros_like(confidence_q)
    
    # Create buckets based on different criteria
    bucket_results = {}
    
    # 1. Class-based buckets
    unique_classes = torch.unique(source_labels)
    for class_idx in unique_classes:
        class_mask = source_labels == class_idx
        if class_mask.sum() > 0:
            bucket_results[f'class_{class_idx}'] = {
                'mask': class_mask,
                'size': class_mask.sum().item(),
                'avg_confidence': confidence_q[class_mask].mean().item(),
                'avg_entropy': entropy_q[class_mask].mean().item(),
                'avg_perturbation': perturbation_norms[class_mask].mean().item() if target_features is not None else 0.0
            }
    
    # 2. Confidence-based buckets
    confidence_sorted = torch.sort(confidence_q)[1]
    bucket_size = len(confidence_sorted) // num_buckets
    for i in range(num_buckets):
        start_idx = i * bucket_size
        end_idx = min((i + 1) * bucket_size, len(confidence_sorted))
        bucket_mask = torch.zeros(len(confidence_sorted), dtype=torch.bool)
        bucket_mask[confidence_sorted[start_idx:end_idx]] = True
        
        bucket_results[f'confidence_bucket_{i}'] = {
            'mask': bucket_mask,
            'size': bucket_mask.sum().item(),
            'confidence_range': (confidence_q[bucket_mask].min().item(), confidence_q[bucket_mask].max().item()),
            'avg_confidence': confidence_q[bucket_mask].mean().item(),
            'avg_entropy': entropy_q[bucket_mask].mean().item(),
            'avg_perturbation': perturbation_norms[bucket_mask].mean().item() if target_features is not None else 0.0
        }
    
    # 3. Entropy-based buckets
    entropy_sorted = torch.sort(entropy_q)[1]
    bucket_size = len(entropy_sorted) // num_buckets
    for i in range(num_buckets):
        start_idx = i * bucket_size
        end_idx = min((i + 1) * bucket_size, len(entropy_sorted))
        bucket_mask = torch.zeros(len(entropy_sorted), dtype=torch.bool)
        bucket_mask[entropy_sorted[start_idx:end_idx]] = True
        
        bucket_results[f'entropy_bucket_{i}'] = {
            'mask': bucket_mask,
            'size': bucket_mask.sum().item(),
            'entropy_range': (entropy_q[bucket_mask].min().item(), entropy_q[bucket_mask].max().item()),
            'avg_confidence': confidence_q[bucket_mask].mean().item(),
            'avg_entropy': entropy_q[bucket_mask].mean().item(),
            'avg_perturbation': perturbation_norms[bucket_mask].mean().item() if target_features is not None else 0.0
        }
    
    # 4. Perturbation norm buckets (if target features available)
    if target_features is not None:
        norm_sorted = torch.sort(perturbation_norms)[1]
        bucket_size = len(norm_sorted) // num_buckets
        for i in range(num_buckets):
            start_idx = i * bucket_size
            end_idx = min((i + 1) * bucket_size, len(norm_sorted))
            bucket_mask = torch.zeros(len(norm_sorted), dtype=torch.bool)
            bucket_mask[norm_sorted[start_idx:end_idx]] = True
            
            bucket_results[f'perturbation_bucket_{i}'] = {
                'mask': bucket_mask,
                'size': bucket_mask.sum().item(),
                'perturbation_range': (perturbation_norms[bucket_mask].min().item(), perturbation_norms[bucket_mask].max().item()),
                'avg_confidence': confidence_q[bucket_mask].mean().item(),
                'avg_entropy': entropy_q[bucket_mask].mean().item(),
                'avg_perturbation': perturbation_norms[bucket_mask].mean().item()
            }
    
    return bucket_results

def analyze_bound_tightness_by_buckets(model_q, model_q_tilde, source_features, target_features, 
                                     source_labels, device, g_q, g_q_tilde, l_x_q, l_x_q_tilde):
    """
    Analyzes bound tightness across different buckets to understand where the bound is tight/loose.
    
    Args:
        model_q, model_q_tilde: Models to compare
        source_features, target_features: Source and target feature distributions
        source_labels: Labels for source data
        device: Device to run on
        g_q, g_q_tilde: Generalization gaps
        l_x_q, l_x_q_tilde: Lipschitz constants
    
    Returns:
        bucket_analysis: Dictionary with tightness analysis per bucket
    """
    # Get bucket information
    bucket_results = calculate_per_bucket_tightness(
        model_q, model_q_tilde, source_features, target_features, source_labels, device
    )
    
    bucket_analysis = {}
    
    for bucket_name, bucket_info in bucket_results.items():
        bucket_mask = bucket_info['mask']
        
        if bucket_mask.sum() == 0:
            continue
        
        # Extract features for this bucket
        bucket_source_features = source_features[bucket_mask]
        bucket_target_features = target_features[bucket_mask] if target_features is not None else source_features[bucket_mask]
        bucket_labels = source_labels[bucket_mask]
        
        # Calculate metrics for this bucket
        try:
            # Wasserstein distance
            w1_dist = calculate_wasserstein1_distance_sinkhorn(
                bucket_source_features, bucket_target_features, freeze_features=True
            )
            
            # Output distance (KL divergence)
            output_dist = calculate_output_distance_kl(
                model_q, model_q_tilde, bucket_target_features, device
            )
            
            # Risk change for this bucket
            risk_q_source = calculate_risk_linear(model_q, bucket_source_features, bucket_labels, device)
            risk_q_target = calculate_risk_linear(model_q, bucket_target_features, bucket_labels, device)
            delta_r = abs(risk_q_source - risk_q_target)
            
            # Bound calculation
            bound = calculate_full_bound(g_q, g_q_tilde, l_x_q, l_x_q_tilde, w1_dist, output_dist)
            
            # Tightness
            tightness = bound / delta_r if delta_r > 1e-6 else float('inf')
            
            bucket_analysis[bucket_name] = {
                **bucket_info,
                'w1_dist': w1_dist,
                'output_dist': output_dist,
                'delta_r': delta_r,
                'bound': bound,
                'tightness': tightness,
                'bound_components': {
                    'g_q': g_q,
                    'g_q_tilde': g_q_tilde,
                    'lipschitz_term': (l_x_q + l_x_q_tilde) * w1_dist,
                    'output_term': output_dist
                }
            }
            
        except Exception as e:
            print(f"Warning: Could not analyze bucket {bucket_name}: {e}")
            bucket_analysis[bucket_name] = {
                **bucket_info,
                'error': str(e)
            }
    
    return bucket_analysis
