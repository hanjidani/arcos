import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel

# --- Data Generation Functions ---
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

def run_mmd_error_experiment(n_samples=1000, alpha=1.0, n_test_samples=10000):
    """
    Runs a single experiment for the MMD-based bound with an explicit error term.
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

    # 3. Calculate MMD-based Bound with Error Term
    empirical_mmd = calculate_mmd(source_data, target_data)
    error_term = 1 / np.sqrt(n_samples)
    
    mmd_bound_with_error = g_q + g_q_tilde + D_Q + empirical_mmd + error_term

    return {
        'alpha': alpha,
        'delta_r': delta_r,
        'g_q': g_q,
        'g_q_tilde': g_q_tilde,
        'D_Q': D_Q,
        'empirical_mmd': empirical_mmd,
        'error_term': error_term,
        'mmd_bound_with_error': mmd_bound_with_error,
        'bound_holds': delta_r <= mmd_bound_with_error
    }

if __name__ == '__main__':
    results = []
    alpha_values = np.linspace(0, 3, 11)
    
    for alpha in alpha_values:
        print(f"Running MMD with error term experiment for alpha = {alpha:.2f}...")
        results.append(run_mmd_error_experiment(alpha=alpha))
        
    df = pd.DataFrame(results)
    pd.set_option('display.width', 200)
    print("\n--- MMD-based Bound with Error Term Results ---")
    print(df)
    
    df.to_csv('exp_3_mmd_bound/mmd_error_term_results.csv', index=False)
    print("\nResults saved to exp_3_mmd_bound/mmd_error_term_results.csv")
