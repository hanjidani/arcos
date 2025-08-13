import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances

# --- Non-Linear Boundary (Fixed MLP) ---
FIXED_W1 = np.random.randn(10, 20)
FIXED_B1 = np.random.randn(20)
FIXED_W2 = np.random.randn(20, 1)
FIXED_B2 = np.random.randn(1)

def true_decision_function(X):
    h = np.maximum(0, X @ FIXED_W1 + FIXED_B1)
    logits = h @ FIXED_W2 + FIXED_B2
    return logits.flatten()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# --- Data Generation ---
def generate_gmm_data(n_samples, n_dim=10):
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(np.random.randn(100, n_dim))
    gmm.means_[1] = gmm.means_[0] + 2.0
    gmm.means_[2] = gmm.means_[0] - 2.0
    X, _ = gmm.sample(n_samples)
    return X, gmm

def apply_additive_noise_shift(data, noise_std_dev):
    """Adds Gaussian noise to the data to create the target distribution."""
    noise = np.random.normal(0, noise_std_dev, data.shape)
    return data + noise

def generate_noisy_labels(X):
    true_logits = true_decision_function(X)
    true_probs = sigmoid(true_logits)
    distance_from_boundary = np.abs(true_probs - 0.5)
    noise_prob = 0.4 * (1 - 2 * distance_from_boundary)
    should_flip = np.random.binomial(1, noise_prob).astype(bool)
    initial_labels = np.random.binomial(1, true_probs)
    final_labels = initial_labels
    final_labels[should_flip] = 1 - final_labels[should_flip]
    return final_labels

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
    X_test, _ = gmm.sample(n_test_samples)
    y_test = generate_noisy_labels(X_test)
    return calculate_risk(model, X_test, y_test)

def run_noise_shift_experiment(n_samples=1500, noise_std_dev=0.5, n_test_samples=10000):
    """
    Runs a single experiment for the MMD-based bound under additive noise shift.
    """
    # 1. Data Generation
    source_data, source_gmm = generate_gmm_data(n_samples)
    source_labels = generate_noisy_labels(source_data)
    
    target_data = apply_additive_noise_shift(source_data, noise_std_dev)
    target_labels = generate_noisy_labels(target_data)
    
    # We need the true target GMM to calculate the generalization gap, which is non-trivial.
    # For this experiment, we will approximate the target GMM by fitting a new GMM to the noisy data.
    target_gmm = GaussianMixture(n_components=3).fit(target_data)


    # 2. Model Training
    source_model = MLPClassifier(hidden_layer_sizes=(20,), max_iter=500, random_state=42).fit(source_data, source_labels)
    target_model = MLPClassifier(hidden_layer_sizes=(20,), max_iter=500, random_state=42).fit(target_data, target_labels)

    # 3. Calculate Risks and Bound Components
    risk_q = calculate_true_risk(source_model, source_gmm)
    risk_q_tilde = calculate_true_risk(target_model, source_gmm)
    delta_r = np.abs(risk_q - risk_q_tilde)

    emp_risk_q = calculate_risk(source_model, source_data, source_labels)
    g_q = np.abs(risk_q - emp_risk_q)
    
    risk_p_tilde_q_tilde = calculate_true_risk(target_model, target_gmm)
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
        'noise_std_dev': noise_std_dev,
        'delta_r': delta_r,
        'mmd_dist': empirical_mmd,
        'mmd_bound': mmd_bound,
        'bound_holds': delta_r <= mmd_bound
    }

if __name__ == '__main__':
    results = []
    noise_levels = np.linspace(0, 2.0, 11)
    
    for noise in noise_levels:
        print(f"Running noise shift experiment for noise_std_dev = {noise:.2f}...")
        results.append(run_noise_shift_experiment(noise_std_dev=noise))
        
    df = pd.DataFrame(results)
    pd.set_option('display.width', 200)
    print("\n--- MMD-based Bound (Additive Noise Shift) Results ---")
    print(df)
    
    df.to_csv('exp_5_noise_shift/noise_shift_results.csv', index=False)
    print("\nResults saved to exp_5_noise_shift/noise_shift_results.csv")
