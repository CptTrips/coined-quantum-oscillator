import unittest
from algebra import condition_subsystem, projector
import numpy as np


class TestConditionSubsystem(unittest.TestCase):

    # Test kron'd example
    def test_example(self):

        I = np.eye(2)
        X = np.array([[0,1],[1,0]])
        O = np.kron(projector(2,0), I) + np.kron(projector(2,1), X)

        v_0 = np.array([1,0])
        v_1 = np.array([0,1])

        self.assertTrue(np.all((condition_subsystem(O, v_0) == I) * (condition_subsystem(O, v_1) == X)))

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

    # Check vector is unit or raises
    def test_non_unit_vector(self):

        O = np.eye(2)

        v = np.array([1,1])

        self.assertRaises(ValueError, test_non_unit_vector, O, v)
