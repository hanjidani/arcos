import numpy as np
from sklearn.mixture import GaussianMixture

# --- Non-Linear Boundary (Fixed MLP) ---
# We define a fixed, simple 2-layer MLP with numpy to act as our "ground truth"
# This is not a trainable model, but a fixed function.
FIXED_W1 = np.random.randn(10, 20)
FIXED_B1 = np.random.randn(20)
FIXED_W2 = np.random.randn(20, 1)
FIXED_B2 = np.random.randn(1)

def true_decision_function(X):
    """A fixed, non-linear function to generate true labels."""
    # First layer with ReLU activation
    h = np.maximum(0, X @ FIXED_W1 + FIXED_B1)
    # Second layer (output logit)
    logits = h @ FIXED_W2 + FIXED_B2
    return logits.flatten()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# --- Data Generation ---
def generate_gmm_data(n_samples, n_dim=10):
    """Generates high-dimensional data from a Gaussian Mixture Model."""
    # Define 3 components for our GMM
    gmm = GaussianMixture(n_components=3, random_state=42)
    # Fit it to some random data to initialize its parameters (means, covariances)
    gmm.fit(np.random.randn(100, n_dim))
    
    # Manually set some means to make it more interesting
    gmm.means_[1] = gmm.means_[0] + 2.0
    gmm.means_[2] = gmm.means_[0] - 2.0

    X, _ = gmm.sample(n_samples)
    return X, gmm

def apply_structured_shift(gmm_params, shift_mean_idx=1, shift_cov_idx=2):
    """Applies a structured shift to the GMM parameters."""
    new_params = {
        'means': gmm_params.means_.copy(),
        'covariances': gmm_params.covariances_.copy(),
        'weights': gmm_params.weights_.copy()
    }
    # Shift the mean of one component
    new_params['means'][shift_mean_idx] += 1.5
    
    # Rotate the covariance of another component
    cov = new_params['covariances'][shift_cov_idx]
    # Simple 2D rotation on the first two dimensions for demonstration
    theta = np.pi / 4 # 45 degrees
    rot_matrix = np.eye(cov.shape[0])
    rot_matrix[0, 0] = np.cos(theta)
    rot_matrix[0, 1] = -np.sin(theta)
    rot_matrix[1, 0] = np.sin(theta)
    rot_matrix[1, 1] = np.cos(theta)
    new_params['covariances'][shift_cov_idx] = rot_matrix @ cov @ rot_matrix.T

    # Create a new GMM with the shifted parameters
    shifted_gmm = GaussianMixture(n_components=3)
    shifted_gmm.weights_ = new_params['weights']
    shifted_gmm.means_ = new_params['means']
    shifted_gmm.covariances_ = new_params['covariances']
    
    return shifted_gmm

# --- Noise Generation ---
def generate_noisy_labels(X):
    """Generates binary labels with feature-dependent noise."""
    true_logits = true_decision_function(X)
    true_probs = sigmoid(true_logits)
    
    # Calculate noise probability: higher noise near the boundary (prob=0.5)
    distance_from_boundary = np.abs(true_probs - 0.5)
    # Noise is highest when distance is lowest. Max noise is e.g. 0.4
    noise_prob = 0.4 * (1 - 2 * distance_from_boundary) 
    
    # Decide which labels to flip
    should_flip = np.random.binomial(1, noise_prob).astype(bool)
    
    # Generate initial labels from true probabilities
    initial_labels = np.random.binomial(1, true_probs)
    
    # Flip the selected labels
    final_labels = initial_labels
    final_labels[should_flip] = 1 - final_labels[should_flip]
    
    return final_labels
