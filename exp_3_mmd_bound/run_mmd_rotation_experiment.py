import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances

# --- Data Generation Functions ---
def generate_base_data(n_samples, mu=[0, 0], sigma=[[1, 0], [0, 1]]):
    return np.random.multivariate_normal(mu, sigma, n_samples)

def apply_rotation_shift(data, theta):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    return data @ rotation_matrix.T

def generate_labels(data, w, b):
    logits = data @ w + b
    probabilities = 1 / (1 + np.exp(-logits))
    return np.random.binomial(1, probabilities)

# --- Core Calculation Functions ---
def calculate_risk(model, data, labels):
    return 1 - model.score(data, labels)

def calculate_mmd(X, Y, gamma):
    K_XX = rbf_kernel(X, X, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)
    K_XY = rbf_kernel(X, Y, gamma)
    mmd2 = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return np.sqrt(mmd2) if mmd2 > 0 else 0

def run_mmd_rotation_experiment(n_samples=1000, theta=0.0, n_test_samples=10000):
    """
    Runs a single experiment for the MMD-based bound under rotation shift
    using the median heuristic for gamma.
    """
    # 1. Data Generation & Model Training
    true_w = np.array([0.5, -0.5])
    true_b = 0.1
    source_data = generate_base_data(n_samples)
    source_labels = generate_labels(source_data, true_w, true_b)
    target_data = apply_rotation_shift(source_data, theta)
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
    test_data_tilde = apply_rotation_shift(generate_base_data(n_test_samples), theta)
    test_labels_tilde = generate_labels(test_data_tilde, true_w, true_b)
    risk_p_tilde_q_tilde = calculate_risk(target_model, test_data_tilde, test_labels_tilde)
    emp_risk_q_tilde = calculate_risk(target_model, target_data, target_labels)
    g_q_tilde = np.abs(risk_p_tilde_q_tilde - emp_risk_q_tilde)
    D_Q = np.abs(emp_risk_q - emp_risk_q_tilde)

    # 3. Calculate MMD-based Bound with Optimal Gamma
    combined_data = np.vstack([source_data, target_data])
    dists = pairwise_distances(combined_data, metric='euclidean')
    median_sq_dist = np.median(dists**2)
    gamma = 1.0 / (2.0 * median_sq_dist)

    empirical_mmd = calculate_mmd(source_data, target_data, gamma=gamma)
    error_term = 1 / np.sqrt(n_samples)
    mmd_bound = g_q + g_q_tilde + D_Q + empirical_mmd + error_term

    return {
        'theta_degrees': np.rad2deg(theta),
        'delta_r': delta_r,
        'optimal_gamma': gamma,
        'mmd_dist': empirical_mmd,
        'mmd_bound': mmd_bound,
        'bound_holds': delta_r <= mmd_bound
    }

if __name__ == '__main__':
    results = []
    # Sweep from 0 to 90 degrees
    theta_values_rad = np.linspace(0, np.pi/2, 11)
    
    for theta in theta_values_rad:
        print(f"Running MMD rotation experiment for theta = {np.rad2deg(theta):.1f} degrees...")
        results.append(run_mmd_rotation_experiment(theta=theta))
        
    df = pd.DataFrame(results)
    pd.set_option('display.width', 200)
    print("\n--- MMD-based Bound (Rotation Shift with Optimal Gamma) ---")
    print(df)
    
    df.to_csv('exp_3_mmd_bound/mmd_rotation_shift_results.csv', index=False)
    print("\nResults saved to exp_3_mmd_bound/mmd_rotation_shift_results.csv")
