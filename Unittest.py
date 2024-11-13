import unittest
import numpy as np
from Mean_Shift_MNIST import CustomMeanShift
from sklearn.datasets import fetch_openml

class TestCustomMeanShift(unittest.TestCase):

    def setUp(self):
        # Załaduj próbkę danych MNIST do testów
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        self.data = mnist.data[:100]  # Używamy 100 próbek dla szybkich testów
        self.model = CustomMeanShift(bandwidth=50, max_iters=10)

    def test_initialization(self):
        """Test inicjalizacji modelu"""
        self.assertEqual(self.model.bandwidth, 50)
        self.assertEqual(self.model.max_iters, 10)
        self.assertIsNone(self.model.centroids)

    def test_fit_method_runs(self):
        """Test czy metoda fit działa bez błędów"""
        try:
            self.model.fit(self.data)
        except Exception as e:
            self.fail(f"Metoda fit wywołała błąd: {e}")

    def test_centroids_not_empty(self):
        """Test czy centroidy nie są puste po dopasowaniu"""
        self.model.fit(self.data)
        self.assertIsNotNone(self.model.centroids)
        self.assertGreater(len(self.model.centroids), 0)

    def test_centroids_convergence(self):
        """Test whether centroids converge during the fitting process"""
        self.model.fit(self.data)
        final_centroids = self.model.centroids

        self.assertIsNotNone(final_centroids)
        self.assertGreater(len(final_centroids), 0)

        self.assertTrue(np.any(np.var(final_centroids, axis=0) > 1e-3))

if __name__ == "__main__":
    unittest.main()

