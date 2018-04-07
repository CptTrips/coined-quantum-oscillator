import numpy as np
from algebra import H


def set_up(periods, coin_op):

    # Prepare basis & initial state
    # Number of position states the particle will access
    particle_state_count = 2*periods + 1

    # Initial spin state
    spin_up_projector = np.array([
        [1, 0],
        [0, 0]
    ], dtype=np.complex128)
    spin_down_projector = np.eye(2) - spin_up_projector

    spin_state = spin_up_projector

    # Initial particle state
    particle_vector_state = np.array([0]* periods + [1] + [0] * periods, dtype=np.complex128)
    particle_state = np.outer(particle_vector_state, particle_vector_state)

    state = np.kron(spin_state, particle_state)

    # Full coin operator
    full_coin_op = np.kron(coin_op, np.eye(particle_state_count))

    # Translation operator
    shift_op = np.zeros((particle_state_count, particle_state_count))
    for i in range(particle_state_count):
        shift_op[i][i-1] = 1

    # Controlled-translation operator
    up_projector = np.kron(spin_up_projector, np.eye(particle_state_count))
    cshiftr_op = up_projector + np.kron(spin_down_projector, shift_op)

    cshiftl_op = cshiftr_op.T

    shift_ops = [cshiftr_op, cshiftl_op]

    return state, full_coin_op, shift_ops


def quantum_walk(state, coin_op, shift_ops, periods):
    """Calculates the state of a harmonic oscillator undergoing a quantum walk protocol.

    Args:
        state (nxn ndarray): Initial state of the oscillator. kron(spin, position).
        coin_op (nxn ndarray): Coin operator
        shift_ops (2xnxn ndarray): Shift operators
        periods (int): Number of oscillator periods to apply the walk protocol for.

    Return:
        nxn ndarray: Final state of the walker
    """

    # Iterate over walk steps
    for i in range(periods):

        for j in range(2):

            # apply coin
            state = coin_op @  state @  H(coin_op)

            # apply shift j
            state = shift_ops[j] @ state @ H(shift_ops[j])

    return state


def simulate(coin_op, periods):

    initial_state, full_coin_op, shift_ops = set_up(periods, coin_op)

    final_state = quantum_walk(initial_state, full_coin_op, shift_ops, periods)

    return final_state


