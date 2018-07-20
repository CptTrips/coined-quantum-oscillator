import numpy as np
import cqo.units as units
from cqo.algebra import H, condition_subsystem, postselect, binary_combinations
from itertools import product


def walk_amplitudes(N, coin_op):
    if N < 1:
        raise ValueError('Number of walk steps must be >= 1')

    if coin_op.shape != (2,2):
        raise ValueError('Coin operator must be valid for qubit (2x2)')

    if not np.allclose(coin_op @ H(coin_op), np.eye(2)):
        raise ValueError('Coin operator must be unitary')

    # Build initial state
    R = N // 2 + N % 2
    L = N // 2

    amplitudes = [[1.0,0]]
    amplitudes = np.concatenate((amplitudes, [[0,0]]*R), axis=0)
    if L:
        amplitudes = np.concatenate(([[0,0]]*L, amplitudes), axis=0)

    # Prepare copies of coin operator
    coin_op_r = np.repeat(coin_op[np.newaxis,:,:], L + R + 1, axis=0)

    for i in range(N):

        # Take the active part of the state & coin op

        s = L - i // 2
        e = L + i // 2 + i % 2 + 1

        active_amplitudes = amplitudes[s:e,:].view()
        active_coin_op = coin_op_r[s:e,:,:].view()

        # Coin op

        active_amplitudes[:,:] = np.einsum('nij,nj->ni', active_coin_op, active_amplitudes)

        # Shift R/L

        if i % 2:
            # shift L
            amplitudes[s-1:e-1, 1] = amplitudes[s:e,1]
            amplitudes[e-1,1] = 0
        else:
            # shift R
            amplitudes[s+1:e+1, 1] = amplitudes[s:e, 1]
            amplitudes[s,1] = 0

    return amplitudes


def coherent_state(alpha, res):
    """
    Samples the density matrix of a coherent state
    """

    # Find cutoff
    # Sample gaussian evenly inside cutoff

    x = np.arange(res - res/2)

    return np.exp(-32*(x - alpha)**2 / (res**2))

def final_state(N, x, x_, s, s_, walk_state, spin_state, coin_amplitudes, alpha,
                decoherence_rate, eps = 1e-3):
    """
    Calculates x, x_, s, s_ matrix element for the final state after an N step coined quantum walk.

    Args:
        N (int): Number of steps in the quantum walk
        x, x_ (float): Spatial matrix element to evaluate
        s, s_ (int): Spin matrix element (s, s_ \in {0,1})
        initial_state: Initial density matrix
        coin_amplitudes (2x2 ndarray): Elements of the coin operator
        alpha (float): Magnitude of displacement
        decoherence_rate (float): Decoherence rate

    Returns:
        complex: Matrix element
    """

    final_term = 0

    print(('\nSampling final state'
          '\nx = {0}, x` = {1}\ns = {2}, s` = {3}\n').format(x, x_, s, s_))

    for S, S_ in product(range(s, N+s), range(s_, N+s_)):

        # sum over S, S_

        element = walk_state.sample(x - S*alpha, x_ - S_*alpha)

        if element < eps:
            continue

        new_eps = eps / element
        max_exponent = abs(np.log(new_eps))

        print('rho(x-S\\alpha, x`-S`\\alpha) = {}'.format(element))
        print('Calculating S = {0}, S` = {1} term'.format(S, S_))

        final_term += (element
                       * sum_over_paths(N, S, S_, x, x_, s, s_,
                                        alpha, decoherence_rate,
                                        coin_amplitudes, spin_state,
                                        max_exponent)
                       * np.exp(-decoherence_rate*(x-x_)**2))

    return final_term

def sum_over_paths(N, S, S_, x, x_, s, s_, alpha, decoherence_rate,
                   coin_amplitudes, spin_state, max_exponent):

    spin_strings = binary_combinations(N-1, S-s)
    spin__strings = binary_combinations(N-1, S_-s_)

    print('Iterating over {}x{} spin combinations...'.format(len(spin_strings),len(spin__strings)))

    total = 0. + 0j

    for s_array, s__array in product(spin_strings, spin__strings):

        s_array = np.concatenate(([0], s_array, [s]))
        s__array = np.concatenate(([0], s__array, [s_]))

        # Sum over combinations

        product_term = 1

        exponent = 0

        for i in range(1, N):

            # Product over spins

            # Multiply by decoherence factor
            # pass if heavily suppressed

            d_s_i = (sum(s_array[i:N]) - sum(s__array[i:N])) * alpha

            exponent += -decoherence_rate * (x - x_ - d_s_i)**2

            if abs(exponent) > max_exponent:
                product_term = 0
                break

            # Retrieve the amplitudes

            a = coin_amplitudes[s_array[i]][s_array[i-1]]
            a_conj = coin_amplitudes[s__array[i]][s__array[i-1]].conjugate()

            product_term *= a * a_conj

        else:

            # If there was no decoherence suppression then multiply by the
            # outer amplitudes and the initial spin state

            a = coin_amplitudes[s][s_array[N-1]]
            a_conj = coin_amplitudes[s_][s__array[N-1]].conjugate()

            decoherence = np.exp(exponent)

            initial_spin_state = spin_state.sample(s_array[0], s__array[0])

            product_term *=  a * a_conj * decoherence * initial_spin_state

        total += product_term

    return total

class SpinState:

    def __init__(self, state_matrix):
        self.state_matrix = state_matrix

    def sample(self, s, s_):
        return self.state_matrix[s][s_]

class CoherentState:

    def __init__(self, alpha, m_omega):
        self.alpha = alpha
        self.m_omega = m_omega
        self.N = pow(m_omega / (units.hbar * np.pi), 0.25)

        self.sigma = np.sqrt((2*units.hbar)/m_omega)

        self.width = self.sigma / 2

    def sample(self, x, x_):

        return self.state(x) * self.state(x_).conjugate()

    def state(self, x):

        h = units.hbar

        mw = self.m_omega

        a = self.alpha

        w = self.sigma

        return self.N * np.exp(-(1/w**2)*(x - w*a.real)**2 + 1j*(2/w)*a.imag*x)

class ThermalState:

    def __init__(self, beta, omega, mass):
        self.beta = beta
        self.omega = omega
        self.mass = mass

        t = np.exp(-beta*units.hbar*omega, dtype=np.float64)

        A = np.array([
            [3/2, 1j, -t, -1j*t],
            [0, 1/2, 1j*t, -t],
            [0, 0, 3/2, -1j],
            [0, 0, 0, 1/2]])
        A = A + A.T

        self.A = A

        self.A_1 = np.linalg.inv(A)

        self.detA = np.linalg.det(A)

        self.B = np.sqrt(2*mass*omega/units.hbar) * np.array([1, 1j, 1, -1j])

        # is this necessary?
        if beta*omega*units.hbar < 1e-3:
            self.Z = 1/beta*units.hbar*omega
        else:
            self.Z = 1/(1-t)

        self.N = np.sqrt(mass * omega / (np.pi * units.hbar * self.detA))
        self.N *= 4/self.Z

        self.mw = mass * omega

        self.width = self._width()

    def sample(self, x, x_):

        B = self.B * np.array([x, x, x_, x_])

        return self.N * np.exp(-0.5*self.mw/units.hbar*(x**2 + x_**2) + 0.5 * B.T@self.A_1@B)

    def _width(self):

        # Find the HWHM
        H = self.sample(0,0)

        w_min = 0
        w_max = np.sqrt(units.hbar / self.mw)

        r = self.sample(w_max, w_max) / H

        while r > 0.5:

            w_min = w_max

            w_max *= 2

            r = self.sample(w_max, w_max) / H

        error = 1e-3

        while abs(r - 0.5) > error:

            w = 0.5 * (w_min + w_max)

            r = self.sample(w, w) / H

            if r > 0.5:
                # w too small
                w_min = w

            elif r < 0.5:
                # w too large
                w_max = w

        return w

def final_state_recursive(N, x, x_, s, s_, walk_state,
                          spin_state, coin_amplitudes, alpha, decoherence, eps):


    if N > 0:

        max_exponent = abs(np.log(eps)) # should precompute this

        exponent = -decoherence*(x - x_)**2

        if abs(exponent) > max_exponent:
            print('Term at ({},{},{},{}) is negligable'.format(x,x_,s,s_))
            return 0

        decoherence_factor = np.exp(exponent)

        new_eps = eps / decoherence_factor

        terms = [final_state_recursive(N-1, x - s*alpha, x_ - s_*alpha, o, o_,
                                       walk_state, spin_state,
                                       coin_amplitudes, alpha, decoherence, new_eps)
                 * coin_amplitudes[s, o] * coin_amplitudes[s_, o_].conjugate()
                 for o, o_ in product([0,1],[0,1])]

        return decoherence_factor * np.sum(terms)

    else:
        return (walk_state.sample(x, x_) * spin_state.sample(s, s_)
                * np.exp(-decoherence*(x-x_)**2))
