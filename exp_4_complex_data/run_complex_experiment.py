import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances

from .complex_data_generation import (
    generate_gmm_data, 
    apply_structured_shift, 
    generate_noisy_labels, 
    true_decision_function, 
    sigmoid
)

# --- Core Calculation Functions ---
def calculate_risk(model, data, labels):
    return 1 - model.score(data, labels)

def calculate_mmd(X, Y, gamma):
    K_XX = rbf_kernel(X, X, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)
    K_XY = rbf_kernel(X, Y, gamma)
    mmd2 = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return np.sqrt(mmd2) if mmd2 > 0 else 0
    
def calculate_true_risk(model, gmm, n_test_samples=10000):
    """Calculates the risk of a model on the true GMM distribution."""
    X_test, _ = gmm.sample(n_test_samples)
    y_test = generate_noisy_labels(X_test)
    return calculate_risk(model, X_test, y_test)


def run_complex_experiment(n_samples=1500):
    """
    Runs a single experiment for the MMD-based bound with complex data.
    """
    # 1. Data Generation
    source_data, source_gmm = generate_gmm_data(n_samples)
    source_labels = generate_noisy_labels(source_data)
    
    shifted_gmm = apply_structured_shift(source_gmm)
    target_data, _ = shifted_gmm.sample(n_samples)
    target_labels = generate_noisy_labels(target_data)

    # 2. Model Training
    # Use MLPClassifiers to learn the non-linear boundary
    source_model = MLPClassifier(hidden_layer_sizes=(20,), max_iter=500, random_state=42).fit(source_data, source_labels)
    target_model = MLPClassifier(hidden_layer_sizes=(20,), max_iter=500, random_state=42).fit(target_data, target_labels)

    # 3. Calculate Risks and Bound Components
    # Note: True risk is now calculated w.r.t. the noisy labeling process
    risk_q = calculate_true_risk(source_model, source_gmm)
    risk_q_tilde = calculate_true_risk(target_model, source_gmm) # On the source distribution P
    delta_r = np.abs(risk_q - risk_q_tilde)

    emp_risk_q = calculate_risk(source_model, source_data, source_labels)
    g_q = np.abs(risk_q - emp_risk_q)
    
    risk_p_tilde_q_tilde = calculate_true_risk(target_model, shifted_gmm)
    emp_risk_q_tilde = calculate_risk(target_model, target_data, target_labels)
    g_q_tilde = np.abs(risk_p_tilde_q_tilde - emp_risk_q_tilde)
    D_Q = np.abs(emp_risk_q - emp_risk_q_tilde)

    # 4. Calculate MMD-based Bound with Optimal Gamma
    combined_data = np.vstack([source_data, target_data])
    dists = pairwise_distances(combined_data, metric='euclidean')
    median_sq_dist = np.median(dists**2)
    gamma = 1.0 / (2.0 * median_sq_dist)

    empirical_mmd = calculate_mmd(source_data, target_data, gamma=gamma)
    error_term = 1 / np.sqrt(n_samples)
    mmd_bound = g_q + g_q_tilde + D_Q + empirical_mmd + error_term

    return {
        'delta_r': delta_r,
        'g_q': g_q,
        'g_q_tilde': g_q_tilde,
        'D_Q': D_Q,
        'mmd_dist': empirical_mmd,
        'mmd_bound': mmd_bound,
        'bound_holds': delta_r <= mmd_bound
    }

if __name__ == '__main__':
    print("Running complex data experiment...")
    result = run_complex_experiment()
    df = pd.DataFrame([result])
    
    pd.set_option('display.width', 200)
    print("\n--- MMD-based Bound (Complex Data) Results ---")
    print(df)
    
    df.to_csv('exp_4_complex_data/complex_results.csv', index=False)
    print("\nResults saved to exp_4_complex_data/complex_results.csv")
