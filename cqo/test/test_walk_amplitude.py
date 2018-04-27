import unittest
from simulation import walk_amplitudes
import numpy as np


class TestWalkAmplitudes(unittest.TestCase):

    I = np.eye(2)
    H = 1/(np.sqrt(2))*np.array([[1,1],[1,-1]])

    def test_N_must_be_positive(self):
        # N < 1 should raise

        for N in range(-1, 1):
            self.assertRaises(ValueError, walk_amplitudes, N)


    def test_identity_coin_op(self):
        # Should return [[0,0]*N/2, [1,0], [0,0]*N/2]

        N = 4

        expected_output = np.array([[0,0]*N/2, [1,0], [0,0]*N/2])

        true_output = walk_amplitudes(N, self.I)

        self.assertTrue(np.all(expected_output == true_output))


    def test_coin_op_must_be_2x2(self):

        self.assertRaises(ValueError, walk_amplitude, 4, np.eye(4))


    def test_coin_op_unitary(self):

        bad_coin_op = np.array([[1,0], [0,0]])

        self.assertRaises(ValueError, walk_amplitude, 4, bad_coin_op)


    def test_output_length(self):

        for i in range(10):

            output = walk_amplitudes(i, self.H)

            self.assertTrue(output.shape == (2, i+1))
