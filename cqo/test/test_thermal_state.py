import unittest
from cqo.simulation import ThermalState, CoherentState
from cqo.units import hbar
import numpy as np
from itertools import product

class TestThermalState(unittest.TestCase):

    eps = 1e-6

    omega = 1e3

    m = 1e-24

    def test_low_T(self):
        """Test that the low temperature behaviour is the ground state
        """

        beta = 1e12 / hbar / self.omega

        rho = ThermalState(beta, self.omega, self.m)

        ground_state = CoherentState(0, self.m*self.omega)

        x_array = np.linspace(-2*rho.width, 2*rho.width)

        for x, x_ in product(x_array, x_array):

            diff = ground_state.sample(x, x_) - rho.sample(x, x_)

            self.assertTrue(abs(diff) < self.eps)

    def beta_range(self):

        return 10.0**np.arange(-2,3) / hbar / self.omega

    def test_width(self):
        """Width property should give half-width-half-maximum
        """

        for beta in self.beta_range():
            rho = ThermalState(beta, self.omega, self.m)

            max = rho.sample(0,0)

            half_max = rho.sample(rho.width, rho.width)

            self.assertTrue(abs(half_max / max - 0.5) < 1e-3)

    def test_norm(self):
        """Test that the state is normalised
        """

        res = 1024

        for beta in self.beta_range():

            rho = ThermalState(beta, self.omega, self.m)

            limit = 8*rho.width

            x_array = np.linspace(-limit, limit, res)

            width = x_array[1] - x_array[0]

            pdf = [rho.sample(x,x) for x in x_array]

            norm = width*sum(pdf)

            if abs(norm - 1) >= self.eps:
                print("beta: {}, norm: {}".format(beta, norm))

            self.assertTrue(abs(norm - 1) < self.eps)

    def test_gaussianity(self):
        """Test that pdf is gaussian across range of temperatures
        """

        for beta in self.beta_range():

            rho = ThermalState(beta, self.omega, self.m)

            x_array = np.linspace(rho.width/10, rho.width, num=9)

            rho_x = np.array([rho.sample(x, x) for x in x_array])

            rho_x /= rho.sample(0,0)

            a = np.log(rho_x) / x_array**2

            self.assertTrue(np.allclose(a / a[0], np.ones(a.shape), self.eps))



if __name__ == "__main__":
    unittest.main()
