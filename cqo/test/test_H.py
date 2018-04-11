import unittest
from algebra import H
import numpy as np

class TestH(unittest.TestCase):


    def random_complex(self, n):
        """Random complex nxn matrix"""
        return np.random.randn(5,5) + 1j*np.random.rand(5,5)


    # Test self-inverse
    def test_self_inverse(self):

        A = self.random_complex(5)

        self.assertTrue(np.all(A == H(H(A))))

    # Test -1 on anti-hermitian
    def test_antihermitian(self):
        A = self.random_complex(5)
        B = A - A.T.conj()

        self.assertTrue(np.all(B == -H(B)))

    # Test identity on Hermitian
    def test_hermitian(self):
        A = self.random_complex(5)

        B = A + A.T.conj()

        self.assertTrue(np.all(B == H(B)))


if __name__ == "__main__":
    unittest.main()
