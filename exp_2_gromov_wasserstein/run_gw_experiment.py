import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import ot

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

def get_loss_matrix(model, data, labels):
    """Computes the pairwise loss difference matrix C[i,j] = |l(i) - l(j)|."""
    # Using log_loss for a smoother, more direct loss value than 0/1 error
    losses = - (labels * model.predict_log_proba(data)[:, 1] + (1 - labels) * model.predict_log_proba(data)[:, 0])
    # Create the pairwise difference matrix
    loss_matrix = np.abs(losses[:, np.newaxis] - losses[np.newaxis, :])
    return loss_matrix

def run_gw_experiment(n_samples=200, alpha=1.0, n_test_samples=10000):
    """
    Runs a single GW experiment. n_samples is kept low as GW is computationally intensive.
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

    # 2. Calculate Risks and Generalization Gaps
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

    # 3. Calculate Gromov-Wasserstein Bound
    C_source = get_loss_matrix(source_model, source_data, source_labels)
    C_target = get_loss_matrix(target_model, target_data, target_labels)
    
    # Uniform weights
    p = ot.unif(n_samples)
    q = ot.unif(n_samples)

    gw_dist = ot.gromov.gromov_wasserstein2(C_source, C_target, p, q, 'square_loss')
    
    gw_bound = g_q + g_q_tilde + gw_dist

    return {
        'alpha': alpha,
        'delta_r': delta_r,
        'g_q': g_q,
        'g_q_tilde': g_q_tilde,
        'gw_dist': gw_dist,
        'gw_bound': gw_bound,
        'bound_holds': delta_r <= gw_bound
    }

if __name__ == '__main__':
    results = []
    alpha_values = np.linspace(0, 3, 11)
    
    for alpha in alpha_values:
        print(f"Running Gromov-Wasserstein experiment for alpha = {alpha:.2f}...")
        results.append(run_gw_experiment(alpha=alpha))
        
    df = pd.DataFrame(results)
    pd.set_option('display.width', 200)
    print("\n--- Gromov-Wasserstein Bound Results ---")
    print(df)
    
    df.to_csv('exp_2_gromov_wasserstein/gw_results.csv', index=False)
    print("\nResults saved to exp_2_gromov_wasserstein/gw_results.csv")

