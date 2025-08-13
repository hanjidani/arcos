import numpy as np
from scipy.stats import wasserstein_distance
import ot

def calculate_wasserstein(dist_a, dist_b):
    """
    Calculates the 1-Wasserstein distance between two 1D distributions.

    Args:
        dist_a (np.ndarray): The first distribution.
        dist_b (np.ndarray): The second distribution.

    Returns:
        float: The 1-Wasserstein distance.
    """
    return wasserstein_distance(dist_a, dist_b)

def calculate_wasserstein_2d(dist_a, dist_b):
    """
    Calculates the 1-Wasserstein distance between two 2D distributions.

    Args:
        dist_a (np.ndarray): The first distribution (n_samples, 2).
        dist_b (np.ndarray): The second distribution (n_samples, 2).

    Returns:
        float: The 1-Wasserstein distance.
    """
    # Cost matrix
    M = ot.dist(dist_a, dist_b)
    # Uniform weights on samples
    a, b = np.ones((dist_a.shape[0],)) / dist_a.shape[0], np.ones((dist_b.shape[0],)) / dist_b.shape[0]
    
    # Earth Mover's Distance
    w1_distance = ot.emd2(a, b, M)
    return w1_distance
