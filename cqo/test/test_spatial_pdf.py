import unittest
from simulate import spatial_pdf
from algebra import projector
import numpy as np

class TestSpatialPdf(unittest.TestCase):

    state = np.eye(10)
    mass = 1
    omega = 1
    alpha_0 = 0
    beta = 1
    error = 1e-6
    resolution = 99

    def test_resolution(self):
        # Test that length of output equals resolution
        P_x, x = spatial_pdf(self.state, self.mass, self.omega, self.alpha_0,
                             self.beta, self.resolution, self.error)

        self.assertEqual(len(P_x), self.resolution)
        self.assertEqual(len(x), self.resolution)

    def test_coherent_state(self):
        # Test that a coherent state produces a gaussian pdf, centered on real
        # part of alpha_0

        idx = 2
        dim = 12
        state = projector(dim, idx)

        P_x, x = spatial_pdf(state, self.mass, self.omega, self.alpha_0,
                             self.beta, self.resolution, self.error)

        n = idx - (dim/2 -1)/2
        gaussian_x = (1/np.pi)**0.5*np.exp(-(x - np.sqrt(2)*n)**2)

        self.assertTrue(np.allclose(P_x, gaussian_x))

    def test_negative_mass(self):
        # Test that negative masses are rejected
        self.assertRaises(ValueError, spatial_pdf, self.state, -1, self.omega,
                     self.alpha_0, self.beta, self.resolution, self.error)

    def test_negative_frequency(self):
        # Test that negative omega is rejected
        self.assertRaises(ValueError, spatial_pdf, self.state, self.mass, -1,
                     self.alpha_0, self.beta, self.resolution, self.error)

    def test_negative_error(self):
        # Test that error must be positive
        self.assertRaises(ValueError, spatial_pdf, self.state, self.mass, self.omega,
                     self.alpha_0, self.beta, self.resolution, -1)


if __name__ == "__main__":
    unittest.main()
