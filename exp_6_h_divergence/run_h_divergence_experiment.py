import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture

# --- Data Generation Functions (from exp_4) ---
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

def generate_gmm_data(n_samples, n_dim=10):
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(np.random.randn(100, n_dim))
    gmm.means_[1] = gmm.means_[0] + 2.0
    gmm.means_[2] = gmm.means_[0] - 2.0
    X, _ = gmm.sample(n_samples)
    return X, gmm

def apply_mean_shift(data, alpha):
    return data + alpha * np.array([1]*data.shape[1])

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

# --- H-Divergence Calculation Functions ---
def calculate_risk(model, data, labels):
    return 1 - model.score(data, labels)

def calculate_empirical_h_divergence(source_data, target_data):
    """Trains a domain classifier to estimate H-divergence."""
    X_combined = np.vstack([source_data, target_data])
    # Domain labels: 0 for source, 1 for target
    y_domain = np.hstack([np.zeros(len(source_data)), np.ones(len(target_data))])
    
    domain_classifier = MLPClassifier(hidden_layer_sizes=(20,), max_iter=500, random_state=42).fit(X_combined, y_domain)
    error = 1 - domain_classifier.score(X_combined, y_domain)
    
    # d_H = 2 * (1 - 2 * error)
    return 2 * (1 - 2 * error)

def calculate_lambda(source_gmm, target_gmm, n_train=1500, n_test=10000):
    """Trains a 'best-effort' model on combined data to estimate lambda."""
    source_train, _ = source_gmm.sample(n_train)
    source_labels = generate_noisy_labels(source_train)
    target_train, _ = target_gmm.sample(n_train)
    target_labels = generate_noisy_labels(target_train)
    
    X_combined = np.vstack([source_train, target_train])
    y_combined = np.hstack([source_labels, target_labels])
    
    best_model = MLPClassifier(hidden_layer_sizes=(50, 20), max_iter=500, random_state=42).fit(X_combined, y_combined)
    
    source_test, _ = source_gmm.sample(n_test)
    source_test_labels = generate_noisy_labels(source_test)
    risk_p = calculate_risk(best_model, source_test, source_test_labels)
    
    target_test, _ = target_gmm.sample(n_test)
    target_test_labels = generate_noisy_labels(target_test)
    risk_p_tilde = calculate_risk(best_model, target_test, target_test_labels)
    
    return risk_p + risk_p_tilde

# --- Main Experiment ---
def run_h_div_experiment(n_samples=1500, alpha=1.0):
    # 1. Data Generation
    source_data, source_gmm = generate_gmm_data(n_samples)
    source_labels = generate_noisy_labels(source_data)
    
    # Create a shifted GMM to represent the true target distribution
    shifted_gmm = GaussianMixture(n_components=3)
    shifted_gmm.weights_ = source_gmm.weights_
    shifted_gmm.means_ = source_gmm.means_ + alpha * np.array([1]*source_data.shape[1])
    shifted_gmm.covariances_ = source_gmm.covariances_
    
    target_data, _ = shifted_gmm.sample(n_samples)
    target_labels = generate_noisy_labels(target_data)

    # 2. Model Training
    source_model = MLPClassifier(hidden_layer_sizes=(20,), max_iter=500, random_state=42).fit(source_data, source_labels)
    target_model = MLPClassifier(hidden_layer_sizes=(20,), max_iter=500, random_state=42).fit(target_data, target_labels)
    
    # 3. Calculate Risks & Gaps
    # (Simplified for clarity - using the GMMs directly)
    risk_q = 1 - source_model.score(source_gmm.sample(10000)[0], generate_noisy_labels(source_gmm.sample(10000)[0]))
    risk_q_tilde = 1 - target_model.score(source_gmm.sample(10000)[0], generate_noisy_labels(source_gmm.sample(10000)[0]))
    delta_r = np.abs(risk_q - risk_q_tilde)
    g_q = np.abs(risk_q - (1 - source_model.score(source_data, source_labels)))
    g_q_tilde = np.abs((1 - target_model.score(shifted_gmm.sample(10000)[0], generate_noisy_labels(shifted_gmm.sample(10000)[0]))) - (1 - target_model.score(target_data, target_labels)))
    D_Q = np.abs((1 - source_model.score(source_data, source_labels)) - (1 - target_model.score(target_data, target_labels)))

    # 4. Calculate H-Divergence Bound Components
    h_divergence = calculate_empirical_h_divergence(source_data, target_data)
    lambda_val = calculate_lambda(source_gmm, shifted_gmm)
    
    h_div_bound = g_q + g_q_tilde + D_Q + 0.5 * h_divergence + lambda_val

    return {
        'alpha': alpha,
        'delta_r': delta_r,
        'h_divergence': h_divergence,
        'lambda': lambda_val,
        'h_div_bound': h_div_bound,
        'bound_holds': delta_r <= h_div_bound
    }

if __name__ == '__main__':
    results = []
    alpha_values = np.linspace(0, 2.0, 11)
    
    for alpha in alpha_values:
        print(f"Running H-Divergence experiment for alpha = {alpha:.2f}...")
        results.append(run_h_div_experiment(alpha=alpha))
        
    df = pd.DataFrame(results)
    pd.set_option('display.width', 200)
    print("\n--- H-Divergence Bound Results ---")
    print(df)
    
    df.to_csv('exp_6_h_divergence/h_divergence_results.csv', index=False)
    print("\nResults saved to exp_6_h_divergence/h_divergence_results.csv")
