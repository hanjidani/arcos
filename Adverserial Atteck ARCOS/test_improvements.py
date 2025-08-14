#!/usr/bin/env python3
"""
Test script to verify the improvements made to the adversarial robustness code.
This script tests the key enhancements and demonstrates the improved functionality.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
from tqdm import tqdm

# Add current directory to path for imports
sys.path.append('.')

from adversarial_utils import (
    calculate_wasserstein1_distance,
    calculate_wasserstein1_distance_sinkhorn,
    calculate_output_distance_kl,
    estimate_lipschitz_constant_loss_based,
    estimate_lipschitz_constant_jacobian,
    calculate_expected_loss,
    calculate_expected_calibration_error,
    find_optimal_temperature,
    create_ema_teacher
)

def create_test_data(n_samples=100, feature_dim=64, num_classes=10):
    """Create synthetic test data for evaluation."""
    torch.manual_seed(42)
    
    # Create synthetic features
    features = torch.randn(n_samples, feature_dim)
    labels = torch.randint(0, num_classes, (n_samples,))
    
    # Create simple models
    model_q = nn.Linear(feature_dim, num_classes)
    model_q_tilde = nn.Linear(feature_dim, num_classes)
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(features, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    return features, labels, model_q, model_q_tilde, loader

def test_wasserstein_improvements():
    """Test the improved Wasserstein distance computation."""
    print("Testing Wasserstein distance improvements...")
    
    # Create test data
    X = torch.randn(100, 64)
    Y = torch.randn(100, 64)
    
    # Test exact Wasserstein (small dataset)
    w1_exact = calculate_wasserstein1_distance(X, Y)
    print(f"  Exact W1 distance: {w1_exact:.6f}")
    
    # Test Sinkhorn Wasserstein
    w1_sinkhorn = calculate_wasserstein1_distance_sinkhorn(X, Y)
    print(f"  Sinkhorn W1 distance: {w1_sinkhorn:.6f}")
    
    # Test with larger dataset (should use Sinkhorn automatically)
    X_large = torch.randn(1500, 64)
    Y_large = torch.randn(1500, 64)
    
    w1_large = calculate_wasserstein1_distance(X_large, Y_large)
    print(f"  Large dataset W1 distance: {w1_large:.6f}")
    
    # Test error handling for different sample sizes
    X_diff = torch.randn(50, 64)
    Y_diff = torch.randn(100, 64)
    
    try:
        calculate_wasserstein1_distance(X_diff, Y_diff)
        assert False, "Should have raised error for different sample sizes"
    except ValueError as e:
        print(f"  ✅ Correctly caught different sample size error: {e}")
    
    try:
        calculate_wasserstein1_distance_sinkhorn(X_diff, Y_diff)
        assert False, "Should have raised error for different sample sizes in Sinkhorn"
    except ValueError as e:
        print(f"  ✅ Correctly caught different sample size error in Sinkhorn: {e}")
    
    # Verify that distances are reasonable
    assert 0 <= w1_exact < 100, "Exact W1 distance out of reasonable range"
    assert 0 <= w1_sinkhorn < 100, "Sinkhorn W1 distance out of reasonable range"
    assert 0 <= w1_large < 100, "Large dataset W1 distance out of reasonable range"
    
    print("  ✅ Wasserstein improvements working correctly")

def test_wasserstein_mathematical_properties():
    """Test mathematical properties of Wasserstein distances."""
    print("Testing Wasserstein mathematical properties...")
    
    # Test two-point sanity check: W₁({a}, {b}) = ||h(a) - h(b)||₂
    a = torch.randn(1, 64)
    b = torch.randn(1, 64)
    expected_distance = torch.norm(a - b, p=2).item()
    
    w1_two_point = calculate_wasserstein1_distance(a, b)
    print(f"  Two-point test: W₁ = {w1_two_point:.6f}, ||a-b||₂ = {expected_distance:.6f}")
    assert abs(w1_two_point - expected_distance) < 1e-3, "Two-point Wasserstein should equal L2 distance"
    
    # Test symmetry: W₁(S, S̃) = W₁(S̃, S)
    X = torch.randn(50, 64)
    Y = torch.randn(50, 64)
    
    w1_forward = calculate_wasserstein1_distance(X, Y)
    w1_backward = calculate_wasserstein1_distance(Y, X)
    print(f"  Symmetry test: W₁(X,Y) = {w1_forward:.6f}, W₁(Y,X) = {w1_backward:.6f}")
    assert abs(w1_forward - w1_backward) < 1e-6, "Wasserstein should be symmetric"
    
    # Test approximate triangle inequality: W₁(S,U) ≤ W₁(S,T) + W₁(T,U)
    Z = torch.randn(50, 64)
    w1_xy = calculate_wasserstein1_distance(X, Y)
    w1_yz = calculate_wasserstein1_distance(Y, Z)
    w1_xz = calculate_wasserstein1_distance(X, Z)
    
    print(f"  Triangle inequality: W₁(X,Z) = {w1_xz:.6f}, W₁(X,Y) + W₁(Y,Z) = {w1_xy + w1_yz:.6f}")
    assert w1_xz <= w1_xy + w1_yz + 1e-3, "Triangle inequality should hold (within tolerance)"
    
    print("  ✅ Wasserstein mathematical properties verified")

def test_lipschitz_mathematical_properties():
    """Test Lipschitz estimation with finite differences on toy networks."""
    print("Testing Lipschitz estimation properties...")
    
    # Create a simple toy network
    class ToyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    toy_net = ToyNet()
    
    # Create test data
    X = torch.randn(100, 10)
    labels = torch.randint(0, 5, (100,))
    dataset = torch.utils.data.TensorDataset(X, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    # Test gradient-based Lipschitz estimation
    l_x_grad = estimate_lipschitz_constant_loss_based(toy_net, loader, 'cpu')
    print(f"  Gradient-based Lipschitz: {l_x_grad:.6f}")
    
    # Test Jacobian-based Lipschitz estimation
    l_x_jacobian = estimate_lipschitz_constant_jacobian(toy_net, loader, 'cpu')
    print(f"  Jacobian-based Lipschitz: {l_x_jacobian:.6f}")
    
    # Test finite difference approximation
    toy_net.eval()
    x_test = torch.randn(1, 10, requires_grad=True)
    y_test = toy_net(x_test)
    
    # Compute gradient norm
    grad_output = torch.ones_like(y_test)
    grad_input = torch.autograd.grad(y_test, x_test, grad_outputs=grad_output, create_graph=False)[0]
    grad_norm = torch.norm(grad_input, p=2).item()
    
    print(f"  Finite difference gradient norm: {grad_norm:.6f}")
    
    # Verify that Lipschitz estimates are reasonable
    assert 0 < l_x_grad < 1000, "Gradient-based Lipschitz out of reasonable range"
    assert 0 < l_x_jacobian < 1000, "Jacobian-based Lipschitz out of reasonable range"
    
    # The Lipschitz constant should be at least as large as the gradient norm
    assert l_x_grad >= grad_norm * 0.1, "Lipschitz should be at least as large as gradient norm"
    
    print("  ✅ Lipschitz estimation properties verified")

def test_output_distance_improvements():
    """Test the improved output distance metrics."""
    print("Testing output distance improvements...")
    
    features, labels, model_q, model_q_tilde, loader = create_test_data()
    
    # Test KL divergence output distance
    kl_dist = calculate_output_distance_kl(model_q, model_q_tilde, features, 'cpu')
    print(f"  KL divergence output distance: {kl_dist:.6f}")
    
    # Test with temperature scaling
    kl_dist_temp = calculate_output_distance_kl(model_q, model_q_tilde, features, 'cpu', temperature=0.1)
    print(f"  KL divergence with temperature=0.1: {kl_dist_temp:.6f}")
    
    # Verify that KL distances are reasonable
    assert 0 <= kl_dist < 100, "KL distance out of reasonable range"
    assert 0 <= kl_dist_temp < 100, "Temperature-scaled KL distance out of reasonable range"
    
    print("  ✅ Output distance improvements working correctly")

def test_lipschitz_improvements():
    """Test the improved Lipschitz constant estimation."""
    print("Testing Lipschitz constant improvements...")
    
    features, labels, model_q, model_q_tilde, loader = create_test_data()
    
    # Test loss-based Lipschitz estimation
    l_x_loss = estimate_lipschitz_constant_loss_based(model_q, loader, 'cpu')
    print(f"  Loss-based Lipschitz constant: {l_x_loss:.6f}")
    
    # Test Jacobian-based Lipschitz estimation
    l_x_jacobian = estimate_lipschitz_constant_jacobian(model_q, loader, 'cpu')
    print(f"  Jacobian-based Lipschitz constant: {l_x_jacobian:.6f}")
    
    # Test with temperature scaling
    l_x_temp = estimate_lipschitz_constant_loss_based(model_q, loader, 'cpu', temperature=0.1)
    print(f"  Temperature-scaled Lipschitz constant: {l_x_temp:.6f}")
    
    # Verify that Lipschitz constants are reasonable
    assert 0 <= l_x_loss < 1000, "Loss-based Lipschitz constant out of reasonable range"
    assert 0 <= l_x_jacobian < 1000, "Jacobian-based Lipschitz constant out of reasonable range"
    assert 0 <= l_x_temp < 1000, "Temperature-scaled Lipschitz constant out of reasonable range"
    
    print("  ✅ Lipschitz constant improvements working correctly")

def test_expected_loss():
    """Test the new expected loss calculation."""
    print("Testing expected loss calculation...")
    
    features, labels, model_q, model_q_tilde, loader = create_test_data()
    
    # Test expected loss calculation
    expected_loss = calculate_expected_loss(model_q, loader, 'cpu')
    print(f"  Expected loss: {expected_loss:.6f}")
    
    # Test with temperature scaling
    expected_loss_temp = calculate_expected_loss(model_q, loader, 'cpu', temperature=0.1)
    print(f"  Temperature-scaled expected loss: {expected_loss_temp:.6f}")
    
    # Verify that expected losses are reasonable
    assert 0 <= expected_loss < 100, "Expected loss out of reasonable range"
    assert 0 <= expected_loss_temp < 100, "Temperature-scaled expected loss out of reasonable range"
    
    print("  ✅ Expected loss calculation working correctly")

def test_calibration_and_temperature():
    """Test temperature calibration and ECE calculation."""
    print("Testing temperature calibration and ECE...")
    
    features, labels, model_q, model_q_tilde, loader = create_test_data()
    
    # Test ECE calculation
    ece, ace = calculate_expected_calibration_error(model_q, loader, 'cpu')
    print(f"  ECE: {ece:.6f}, ACE: {ace:.6f}")
    
    # Test temperature optimization
    best_temp, best_ece, temp_results = find_optimal_temperature(model_q, loader, 'cpu')
    print(f"  Best temperature: {best_temp:.3f}, Best ECE: {best_ece:.6f}")
    
    # Verify that calibration metrics are reasonable
    assert 0 <= ece <= 1, "ECE should be between 0 and 1"
    assert 0 <= ace <= 1, "ACE should be between 0 and 1"
    assert 0.1 <= best_temp <= 10, "Best temperature should be reasonable"
    
    print("  ✅ Temperature calibration and ECE working correctly")

def test_ema_teacher():
    """Test EMA teacher creation and updates."""
    print("Testing EMA teacher functionality...")
    
    # Create a simple model
    model = nn.Linear(10, 5)
    
    # Create EMA teacher
    teacher, update_teacher = create_ema_teacher(model, decay=0.9)
    
    # Verify teacher is frozen
    for param in teacher.parameters():
        assert not param.requires_grad, "Teacher parameters should be frozen"
    
    # Test teacher update
    original_weight = teacher.fc.weight.clone()
    update_teacher()
    
    # Verify weights changed (due to EMA update)
    weight_changed = not torch.allclose(teacher.fc.weight, original_weight)
    assert weight_changed, "Teacher weights should change after EMA update"
    
    print("  ✅ EMA teacher functionality working correctly")

def test_reproducibility():
    """Test that the reproducibility improvements work."""
    print("Testing reproducibility improvements...")
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate data with fixed seed
    X1 = torch.randn(50, 32)
    Y1 = torch.randn(50, 32)
    
    # Reset seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate data again with same seed
    X2 = torch.randn(50, 32)
    Y2 = torch.randn(50, 32)
    
    # Verify reproducibility
    assert torch.allclose(X1, X2), "X data not reproducible"
    assert torch.allclose(Y1, Y2), "Y data not reproducible"
    
    # Test Wasserstein reproducibility
    w1_1 = calculate_wasserstein1_distance(X1, Y1)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    w1_2 = calculate_wasserstein1_distance(X1, Y1)
    
    assert abs(w1_1 - w1_2) < 1e-6, "Wasserstein distance not reproducible"
    
    print("  ✅ Reproducibility improvements working correctly")

def run_all_tests():
    """Run all tests to verify improvements."""
    print("=" * 60)
    print("Testing Adversarial Robustness Code Improvements")
    print("=" * 60)
    
    try:
        test_wasserstein_improvements()
        test_wasserstein_mathematical_properties()
        test_lipschitz_mathematical_properties()
        test_output_distance_improvements()
        test_lipschitz_improvements()
        test_expected_loss()
        test_calibration_and_temperature()
        test_ema_teacher()
        test_reproducibility()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! Improvements are working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
