import numpy as np

def generate_base_data(n_samples, mu=[0, 0], sigma=[[1, 0], [0, 1]]):
    """
    Generates the base 2D Gaussian data.

    Args:
        n_samples (int): The number of samples to generate.
        mu (list, optional): The mean of the distribution. Defaults to [0, 0].
        sigma (list, optional): The covariance matrix. Defaults to [[1, 0], [0, 1]].

    Returns:
        np.ndarray: The generated data.
    """
    return np.random.multivariate_normal(mu, sigma, n_samples)

def apply_mean_shift(data, alpha, direction=[1, 1]):
    """
    Applies a mean shift to the data.

    Args:
        data (np.ndarray): The input data.
        alpha (float): The shift severity.
        direction (list, optional): The direction of the shift. Defaults to [1, 1].

    Returns:
        np.ndarray: The shifted data.
    """
    shift_vector = alpha * np.array(direction)
    return data + shift_vector

def apply_covariance_shift(data, beta):
    """
    Applies a covariance shift (anisotropy) to the data.

    Args:
        data (np.ndarray): The input data.
        beta (float): The anisotropy factor.

    Returns:
        np.ndarray: The shifted data.
    """
    covariance_matrix = np.array([[beta, 0], [0, 1/beta]])
    # We apply the transformation to the data points directly.
    # This is equivalent to transforming the covariance matrix of the generating distribution.
    return data @ covariance_matrix.T

def apply_rotation_shift(data, theta):
    """
    Applies a rotation shift to the data.

    Args:
        data (np.ndarray): The input data.
        theta (float): The rotation angle in radians.

    Returns:
        np.ndarray: The shifted data.
    """
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    return data @ rotation_matrix.T

def generate_labels(data, w, b):
    """
    Generates binary labels based on a linear decision boundary.

    Args:
        data (np.ndarray): The input data.
        w (np.ndarray): The weight vector.
        b (float): The bias term.

    Returns:
        np.ndarray: The binary labels (0 or 1).
    """
    logits = data @ w + b
    probabilities = 1 / (1 + np.exp(-logits))
    return np.random.binomial(1, probabilities)
