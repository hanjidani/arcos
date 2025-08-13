import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .data_generation import generate_base_data, apply_covariance_shift, generate_labels
from src.wasserstein import calculate_wasserstein_2d

def calculate_risk(model, data, labels):
    """Calculates the risk (1 - accuracy) of a model."""
    return 1 - model.score(data, labels)

def calculate_lipschitz_data_dependent(model, data, labels, percentile=100):
    """Calculates a data-dependent Lipschitz constant for the loss w.r.t. inputs."""
    w = model.coef_[0]
    b = model.intercept_[0]
    logits = data @ w + b
    probabilities = 1 / (1 + np.exp(-logits))
    grad_z_l = probabilities - labels
    grad_x_l = grad_z_l[:, np.newaxis] * w
    grad_norms = np.linalg.norm(grad_x_l, axis=1)
    return np.percentile(grad_norms, percentile)

def run_covariance_experiment(n_samples=1000, beta=1.0, n_test_samples=10000):
    """
    Runs a single experiment for the covariance shift scenario.
    """
    # 1. Data Generation
    true_w = np.array([0.5, -0.5])
    true_b = 0.1
    source_data = generate_base_data(n_samples)
    source_labels = generate_labels(source_data, true_w, true_b)
    # Apply covariance shift
    target_data = apply_covariance_shift(source_data, beta)
    target_labels = generate_labels(target_data, true_w, true_b)

    # 2. Model Training
    source_model = LogisticRegression().fit(source_data, source_labels)
    target_model = LogisticRegression().fit(target_data, target_labels)

    # 3. Calculate Core Components of Bound 1
    test_data = generate_base_data(n_test_samples)
    test_labels = generate_labels(test_data, true_w, true_b)
    risk_q = calculate_risk(source_model, test_data, test_labels)
    risk_q_tilde = calculate_risk(target_model, test_data, test_labels)
    delta_r = np.abs(risk_q - risk_q_tilde)
    emp_risk_q = calculate_risk(source_model, source_data, source_labels)
    g_q = np.abs(risk_q - emp_risk_q)
    
    test_data_tilde = apply_covariance_shift(generate_base_data(n_test_samples), beta)
    test_labels_tilde = generate_labels(test_data_tilde, true_w, true_b)
    risk_p_tilde_q_tilde = calculate_risk(target_model, test_data_tilde, test_labels_tilde)
    emp_risk_q_tilde = calculate_risk(target_model, target_data, target_labels)
    g_q_tilde = np.abs(risk_p_tilde_q_tilde - emp_risk_q_tilde)
    
    w1_dist = calculate_wasserstein_2d(source_data, target_data)
    D_Q = np.abs(emp_risk_q - emp_risk_q_tilde)
    
    # Using 99th percentile Lipschitz for our best bound
    lipschitz_99pct = calculate_lipschitz_data_dependent(target_model, test_data_tilde, test_labels_tilde, percentile=99)
    shift_penalty_99pct = lipschitz_99pct * w1_dist
    bound_99pct = g_q + g_q_tilde + D_Q + shift_penalty_99pct

    return {
        'beta': beta,
        'delta_r': delta_r,
        'g_q': g_q,
        'g_q_tilde': g_q_tilde,
        'D_Q': D_Q,
        'w1_dist': w1_dist,
        'lipschitz_99pct': lipschitz_99pct,
        'shift_penalty_99pct': shift_penalty_99pct,
        'bound_99pct': bound_99pct,
        'bound_holds': delta_r <= bound_99pct
    }

if __name__ == '__main__':
    results = []
    beta_values = np.linspace(0.5, 2.0, 11)
    
    for beta in beta_values:
        print(f"Running covariance shift experiment for beta = {beta:.2f}...")
        results.append(run_covariance_experiment(beta=beta))
        
    df = pd.DataFrame(results)
    pd.set_option('display.width', 200)
    print("\n--- Covariance Shift Bound 1 (99pct) Results ---")
    print(df)
    
    df.to_csv('exp_1_bound_tightness/covariance_shift_results.csv', index=False)
    print("\nResults saved to exp_1_bound_tightness/covariance_shift_results.csv")
