import unittest
import numpy as np
from src.wasserstein import calculate_wasserstein

class TestWasserstein(unittest.TestCase):

    def test_calculate_wasserstein(self):
        dist_a = np.array([1, 2, 3, 4, 5])
        dist_b = np.array([6, 7, 8, 9, 10])
        distance = calculate_wasserstein(dist_a, dist_b)
        self.assertAlmostEqual(distance, 5.0)

if __name__ == '__main__':
    unittest.main()
