import unittest
from algebra import condition_subsystem, projector
import numpy as np


class TestConditionSubsystem(unittest.TestCase):

    # Test kron'd example
    def test_example(self):

        p_0 = p_1 = .5

        state_0 = .5 * np.eye(2)
        state_1 = .5 * (np.eye(2) + np.array([[0,1],[1,0]]))
        O = p_0 * np.kron(projector(2,0), state_0) + p_1 * np.kron(projector(2,1), state_1)

        v_0 = np.array([1,0])
        v_1 = np.array([0,1])

        O_A_0 = condition_subsystem(O, v_0)
        O_A_1 = condition_subsystem(O, v_1)

        def normalise(O):
            return O / np.trace(O)

        O_A_0 = normalise(O_A_0)

        O_A_1 = normalise(O_A_1)

        self.assertTrue(np.all((O_A_0 == state_0) * (O_A_1 == state_1)))


    # Check they factor or raises
    def test_non_factoring(self):

        O = np.eye(8)

        v = np.array([1, 0, 0])

        self.assertRaises(ValueError, condition_subsystem, O, v)

    # Check they're square or raises
    def test_non_square_O(self):
        
        O = np.ones((4,6))

        v = np.array([1,0])
        
        self.assertRaises(ValueError, condition_subsystem, O, v)

