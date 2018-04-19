import unittest
from algebra import tensor_swap
import numpy as np

class TestTensorSwap(unittest.TestCase):

    # Test non-factoring dim_A
    def test_non_factoring_dim_A(self):
        O = np.eye(5)
        self.assertRaises(ValueError, tensor_swap. O, 2)

    # Test kron'd example
    def test_tensor_product(self):
        X = np.array([[0,1],[1,0]])
        O = np.kron(np.eye(2), X)
        P = np.kron(X, np.eye(2))

        self.assertTrue(np.all(tensor_swap(O, 2) == P))

        self.assertTrue(np.all(tensor_swap(P, 2) == O))

    # Test 2 <= dim_A <= len(O) / 2
    def test_dim_A_larger_2(self):

        O = np.eye(4)
       
        for i in range(3, -1):

            self.assertRaises(ValueError, tensor_swap, O, i)

        self.assertTrue(np.all(tensor_swap(O, 2) == np.eye(4)))

    def test_dim_A_less_half_O(self):
        
        # We expect all but the edge case (dim_A == size(O)) to be caught by
        # non-factoring

        for i in range(4,11,2):
            O = np.eye(i)

            self.assertRaises(ValueError, tensor_swap, O, i)

    # Test non-square O raises
    def test_non_square_O(self):

        O = np.ones((4,6))

        self.assertRaises(ValueError, tensor_swap, O, 2)
