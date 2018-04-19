import unittest
from algebra import projector
import numpy as np


class TestProjector(unittest.TestCase):

    dim = 4

    # Test P^2 = P
    def test_P_squared(self):

        P = projector(10, 5)

        self.assertTrue(np.all(P@P == P))

    # Test PQ = delta_PQ P
    def test_pq_delta(self):

        i_array = np.arange(self.dim)

        P = [projector(self.dim, i) for i in i_array]

        for i, j in zip(i_array, i_array):

            if i == j:

                self.assertTrue(np.all(P[i]@P[j] == P[i]))

            else:

                self.assertTrue(np.all(P[i] @ P[j] == np.zeros((self.dim, self.dim))))


    # Test sum_P P = I
    def test_sum_P_is_I(self):

        sum_P = np.zeros((self.dim,self.dim), dtype=np.complex128)

        for i in range(self.dim):

            sum_P += projector(self.dim, i)

        self.assertTrue(np.all(sum_P == np.eye(self.dim)))

    # Test output is dimension dim
    def test_output_shape(self):

        P = projector(self.dim, 2)

        self.assertEqual(P.shape, (self.dim,self.dim))

    # Test raise on i > dim
    def test_i_greater_dim_raises(self):

        self.assertRaises(ValueError, projector, self.dim, self.dim)

    def test_negative_i_raises(self):
        
        self.assertRaises(ValueError, projector, self.dim, -1)


    def test_negative_dim_raises(self):

        self.assertRaises(ValueError, projector, -self.dim, 0)


    def test_dim_0_raises(self):

        self.assertRaises(ValueError, projector, 0, 0)


if __name__ == "__main__":
    unittest.main()
