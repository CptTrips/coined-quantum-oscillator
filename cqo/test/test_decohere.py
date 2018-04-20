import unittest
from simulate import decohere
import numpy as np

class TestDecohere(unittest.TestCase):

    # gamma=0 is identity
    def test_no_decoherence(self):
        state = np.random.randn(10,10)

        final_state = decohere(state, 0)

        self.assertTrue(np.allclose(state, final_state))

    def test_no_negative_gamma(self):
        state = np.eye(6)
        self.assertRaises(ValueError, decohere, state, -1.0)
