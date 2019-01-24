from cqo.simulation import final_state, SpinState, CoherentState
from cqo import units
from cqo import output
import numpy as np


def run_model():

    N = 10

    decoherence_rate = 1

    alpha = 1

    # Create a SpinState and a Coherent/Thermal state

    spin_state = SpinState(np.array([[1, 0], [0, 0]])/np.sqrt(2))

    walk_state = CoherentState(0, units.hbar*8)

    # Create the coin operator

    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    coin_op = (sigma_x + sigma_z)/np.sqrt(2)

    x = range(-N, N)

    sample = lambda x_i, s : final_state(N, x_i, x_i, s, s, walk_state,
                                         spin_state, coin_op, alpha, decoherence_rate)

    pdf = [[sample(x_i, 0),  sample(x_i, 1)] for x_i in x]

    output.draw_pdf(x, pdf[:,0], pdf[:,1], "CQW Model 1")
