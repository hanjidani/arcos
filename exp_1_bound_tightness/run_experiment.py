import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .data_generation import generate_base_data, apply_mean_shift, generate_labels
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

def calculate_output_distance(model1, model2, data):
    """Calculates the average L2 distance between model predictions."""
    preds1 = model1.predict_proba(data)
    preds2 = model2.predict_proba(data)
    distances = np.linalg.norm(preds1 - preds2, axis=1)
    return np.mean(distances)

def run_definitive_experiment(n_samples=1000, alpha=1.0, n_test_samples=10000):
    """
    Runs a single, definitive experiment to test Bound 1 with different Lipschitz estimates.
    """
    # 1. Data Generation & 2. Model Training (omitted for brevity)
    true_w = np.array([0.5, -0.5])
    true_b = 0.1
    source_data = generate_base_data(n_samples)
    source_labels = generate_labels(source_data, true_w, true_b)
    target_data = apply_mean_shift(source_data, alpha)
    target_labels = generate_labels(target_data, true_w, true_b)
    source_model = LogisticRegression()
    source_model.fit(source_data, source_labels)
    target_model = LogisticRegression()
    target_model.fit(target_data, target_labels)

    # 3. Calculate Core Components
    test_data = generate_base_data(n_test_samples)
    test_labels = generate_labels(test_data, true_w, true_b)
    risk_q = calculate_risk(source_model, test_data, test_labels)
    risk_q_tilde = calculate_risk(target_model, test_data, test_labels)
    delta_r = np.abs(risk_q - risk_q_tilde)
    emp_risk_q = calculate_risk(source_model, source_data, source_labels)
    g_q = np.abs(risk_q - emp_risk_q)
    test_data_tilde = apply_mean_shift(generate_base_data(n_test_samples), alpha)
    test_labels_tilde = generate_labels(test_data_tilde, true_w, true_b)
    risk_p_tilde_q_tilde = calculate_risk(target_model, test_data_tilde, test_labels_tilde)
    emp_risk_q_tilde = calculate_risk(target_model, target_data, target_labels)
    g_q_tilde = np.abs(risk_p_tilde_q_tilde - emp_risk_q_tilde)
    w1_dist = calculate_wasserstein_2d(source_data, target_data)
    D_Q = np.abs(emp_risk_q - emp_risk_q_tilde)

    # --- Bound 1 with Different Lipschitz Estimates ---
    # Max Lipschitz (100th percentile)
    lipschitz_max = calculate_lipschitz_data_dependent(target_model, test_data_tilde, test_labels_tilde, percentile=100)
    shift_penalty_max = lipschitz_max * w1_dist
    bound_max = g_q + g_q_tilde + D_Q + shift_penalty_max

    # 99th Percentile Lipschitz
    lipschitz_99pct = calculate_lipschitz_data_dependent(target_model, test_data_tilde, test_labels_tilde, percentile=99)
    shift_penalty_99pct = lipschitz_99pct * w1_dist
    bound_99pct = g_q + g_q_tilde + D_Q + shift_penalty_99pct

    return {
        'alpha': alpha,
        'delta_r': delta_r,
        'g_q': g_q,
        'g_q_tilde': g_q_tilde,
        'D_Q': D_Q,
        'w1_dist': w1_dist,
        'lipschitz_max': lipschitz_max,
        'shift_penalty_max': shift_penalty_max,
        'bound_max': bound_max,
        'lipschitz_99pct': lipschitz_99pct,
        'shift_penalty_99pct': shift_penalty_99pct,
        'bound_99pct': bound_99pct,
        'bound_holds': delta_r <= bound_99pct
    }

if __name__ == '__main__':
    results = []
    alpha_values = np.linspace(0, 3, 11)
    
    for alpha in alpha_values:
        print(f"Running Bound 1 percentile comparison for alpha = {alpha:.2f}...")
        results.append(run_definitive_experiment(alpha=alpha))
        
    df = pd.DataFrame(results)
    pd.set_option('display.width', 200)
    print("\n--- Bound 1 - Lipschitz Percentile Comparison ---")
    print(df)
    
    df.to_csv('exp_1_bound_tightness/definitive_results_bound1_percentile_comparison.csv', index=False)
    print("\nDebug results saved to exp_1_bound_tightness/definitive_results_bound1_percentile_comparison.csv")
