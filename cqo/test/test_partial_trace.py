import unittest
from algebra import partial_trace, projector
import numpy as np


class TestPartialTrace(unittest.TestCase):

    
    # Raise if they don't factor (also excluding dim_A = {1, len(O))
    def test_factor(self):

        # 5 is our example prime
        O = np.eye(5)

        for i in range(1,6,1):
            self.assertRaises(ValueError, partial_trace, O, i)

    # Raise if O isn't square
    def test_shape(self):

        O = np.ones((4,6))

        self.assertRaises(ValueError, partial_trace, O, 2)

    # Test an example
    def test_example(self):

        I = np.eye(2)
        X = np.array([[0,1],[1,0]])

        O = np.kron(projector(2,0), I) + np.kron(projector(2,1), X)

        IplusX = np.ones((2,2))

        O_trB = partial_trace(O, 2)

        self.assertTrue(np.all(O_trB == IplusX))
