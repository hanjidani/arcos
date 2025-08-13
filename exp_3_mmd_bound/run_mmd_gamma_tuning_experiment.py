import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances

# --- Data Generation Functions ---
# (Omitted for brevity, they are the same as before)
def generate_base_data(n_samples, mu=[0, 0], sigma=[[1, 0], [0, 1]]):
    return np.random.multivariate_normal(mu, sigma, n_samples)
def apply_mean_shift(data, alpha, direction=[1, 1]):
    return data + alpha * np.array(direction)
def generate_labels(data, w, b):
    logits = data @ w + b
    probabilities = 1 / (1 + np.exp(-logits))
    return np.random.binomial(1, probabilities)

# --- Core Calculation Functions ---
def calculate_risk(model, data, labels):
    return 1 - model.score(data, labels)

def calculate_mmd(X, Y, gamma=1.0):
    K_XX = rbf_kernel(X, X, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)
    K_XY = rbf_kernel(X, Y, gamma)
    mmd2 = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return np.sqrt(mmd2) if mmd2 > 0 else 0

def run_gamma_tuning_experiment(gamma, n_samples=1000, alpha=1.5, n_test_samples=10000):
    """
    Runs a single experiment for a given gamma value.
    """
    # 1. Data Generation & Model Training
    true_w = np.array([0.5, -0.5])
    true_b = 0.1
    source_data = generate_base_data(n_samples)
    source_labels = generate_labels(source_data, true_w, true_b)
    target_data = apply_mean_shift(source_data, alpha)
    target_labels = generate_labels(target_data, true_w, true_b)

    source_model = LogisticRegression().fit(source_data, source_labels)
    target_model = LogisticRegression().fit(target_data, target_labels)

    # 2. Calculate Risks and Bound Components
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
    D_Q = np.abs(emp_risk_q - emp_risk_q_tilde)

    # 3. Calculate MMD-based Bound with the given gamma
    empirical_mmd = calculate_mmd(source_data, target_data, gamma=gamma)
    error_term = 1 / np.sqrt(n_samples)
    mmd_bound = g_q + g_q_tilde + D_Q + empirical_mmd + error_term

    return {
        'gamma': gamma,
        'delta_r': delta_r,
        'mmd_dist': empirical_mmd,
        'mmd_bound': mmd_bound,
        'bound_holds': delta_r <= mmd_bound
    }

if __name__ == '__main__':
    # Determine a good gamma using the median heuristic
    temp_data = generate_base_data(1000)
    dists = pairwise_distances(temp_data, metric='euclidean')
    median_sq_dist = np.median(dists**2)
    heuristic_gamma = 1.0 / (2.0 * median_sq_dist)
    print(f"Median Heuristic suggests gamma = {heuristic_gamma:.4f}")

    results = []
    # Test a range of gammas centered around the heuristic
    gamma_values = np.logspace(-2, 2, 11) * heuristic_gamma
    
    for gamma in gamma_values:
        print(f"Running MMD gamma tuning experiment for gamma = {gamma:.4f}...")
        results.append(run_gamma_tuning_experiment(gamma=gamma))
        
    df = pd.DataFrame(results)
    pd.set_option('display.width', 200)
    print("\n--- MMD Gamma Tuning Results (alpha=1.5) ---")
    print(df)
    
    df.to_csv('exp_3_mmd_bound/mmd_gamma_tuning_results.csv', index=False)
    print("\nResults saved to exp_3_mmd_bound/mmd_gamma_tuning_results.csv")
