import unittest
from simulate import set_up
import numpy as np

class SetUpTest(unittest.TestCase):

    hadamard = (1/np.sqrt(2))*np.array([[1,1],[1,-1]])

    def test_negative_period_raises(self):
        rand_negative = np.random.randint(-1e6, high=0)
        self.assertRaises(ValueError, set_up, rand_negative, self.hadamard)

    def test_bad_shape_coin_op_raises(self):
        bad_shape_coin_op = np.eye(3)
        bad_shape_coin_op[0:2,0:2] = self.hadamard
        self.assertRaises(ValueError, set_up, 1, bad_shape_coin_op)

    def test_nonunitary_coin_op_raises(self):
        nonunitary_coin_op = np.array([[1,0],[0,0]])
        self.assertRaises(ValueError, set_up, 1, nonunitary_coin_op)

    def test_output_shape(self):
        periods = 8**np.arange(1,4)
        for p in periods:
            output = set_up(p, self.hadamard)
            expected_shape = (2*(2*(p+1)+1), 2*(2*(p+1)+1))

            output_flat = output[0:1] + (output[2][0], output[2][1])
            for o in output_flat:
                self.assertEqual(o.shape, expected_shape)


    def test_output_state(self):
        """Checks output state is Hermitian, positive & unit trace"""

        state, c, s = set_up(2, self.hadamard)

        # Hermitian
        self.assertTrue(np.allclose(state, state.T.conj()))

        # positive
        self.assertTrue(np.all(np.linalg.eigvalsh(state) >= 0))

        # Unit trace
        self.assertTrue(abs(np.trace(state) - 1) < 1e-5)

    def test_output_operators_unitary(self):

        s, coin_op, shift_ops = set_up(2, self.hadamard)

        operators = np.append([coin_op], shift_ops, axis=0)

        I = np.eye(len(coin_op))

        for o in operators:
            self.assertTrue(np.allclose(o @ o.T.conj(), I))

if __name__ == '__main__':
    unittest.main()
