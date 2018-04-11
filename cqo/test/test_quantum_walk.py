import unittest
from simulate import quantum_walk, set_up
import numpy as np
from algebra import H


class TestQuantumWalk(unittest.TestCase):

    # Test identity returns state
    def test_identity(self):
        state = np.random.rand(6,6)
        coin_op = np.eye(6)
        shift_ops = [np.eye(6), np.eye(6)]
        gamma = 0
        final_state = quantum_walk(state, coin_op, shift_ops, 1, gamma)

        self.assertTrue(np.allclose(state, final_state))

    # Test inverse matrices unwind quantum walk
    def test_inverse(self):

        state = np.random.rand(10,10)

        coin_op = 1/np.sqrt(2)*np.array([[1,1],[1,-1]])

        s, full_coin_op, shift_ops = set_up(1, coin_op)

        final_state = quantum_walk(state, full_coin_op, shift_ops, 1, 0)

        unwind_operator = H(shift_ops[1] @ full_coin_op @ shift_ops[0]
                            @ full_coin_op)

        unwound_state = unwind_operator @ final_state @ H(unwind_operator)

        self.assertTrue(np.allclose(state, unwound_state))


if __name__ == '__main__':
    unittest.main()
