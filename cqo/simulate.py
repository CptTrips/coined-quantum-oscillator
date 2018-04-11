import numpy as np
from algebra import H
from algebra import condition_subsystem


def set_up(periods, coin_op):

    #! Validate periods >= 0

    # Prepare basis & initial state

    # Initial spin state
    spin_up_projector = np.array([
        [1, 0],
        [0, 0]
    ], dtype=np.complex128)
    spin_down_projector = np.eye(2) - spin_up_projector

    spin_state = spin_up_projector

    # Initial particle state
    particle_vector_state = np.array([0] * (periods+1) + [1] + [0] * (periods+1),
                                     dtype=np.complex128)
    particle_state = np.outer(particle_vector_state, particle_vector_state)

    # Number of position states the particle will access
    particle_state_count = len(particle_vector_state)

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


def decohere(state, gamma):
    """Applies spatial decoherence in the coherent state basis.

    Off-diagonal elements of the state will decay exponentially in the site
    distance between states.

    Args:
        state (nxn ndarray): State before decoherence
        gamma (float): Decoherence rate

    Return:
        nxn ndarray: State after decoherence
    """

    # Build decay matrix and apply element-wise to state
    site_idx = np.arange(len(state)/2)
    site_idx_i, site_idx_j = np.meshgrid(site_idx, site_idx.T)
    site_diff = site_idx_i - site_idx_j

    decay = np.exp(-(gamma/2) * site_diff**2)

    decay_full = np.kron(np.ones((2,2)), decay)

    final_state = decay_full * state

    return final_state

def quantum_walk(state, coin_op, shift_ops, periods, gamma):
    """Calculates the state of a harmonic oscillator undergoing a quantum walk protocol.

    Args:
        state (nxn ndarray): Initial state of the oscillator. kron(spin, position).
        coin_op (nxn ndarray): Coin operator
        shift_ops (2xnxn ndarray): Shift operators
        periods (int): Number of oscillator periods to apply the walk protocol for.
        gamma (float): Decoherence rate

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

            # apply decoherence
            if gamma > 0:
                state = decohere(state, gamma)

    return state


def simulate(coin_op, periods, gamma):

    initial_state, full_coin_op, shift_ops = set_up(periods, coin_op)

    final_state = quantum_walk(initial_state, full_coin_op, shift_ops, periods, gamma)

    return final_state

def spatial_pdf(state, mass, omega, alpha_0, beta, resolution, error):
    """Calculates the spatial probability distribution of a density matrix whose basis states
    are evenly spaced coherent states.

    Args:
        state (nxn ndarray): Density matrix in coherent lattice basis
        mass (float): Particle mass
        omega (float): Oscillator angular frequency
        alpha_0 (complex): Initial coherent state
        beta (complex): Lattice displacement
        resolution (int): Number of points to sample
        error (float): Coherent state amplitude must be above this value to
                        contribute to the probability at point x

    Return:
        ndarray: 2xresolution array of [x, prob] pairs
    """


    up_state, p_up = condition_subsystem(state, [1,0])
    down_state, p_down = condition_subsystem(state, [0,1])

    node_count = len(up_state)

    # Get array of positions to sample
    x = (beta.real*np.linspace(-node_count/2, node_count/2, num=resolution)
        - alpha_0.real)

    # Build vector of \psi^{alpha_n}(x)
    n_vector = np.arange(node_count) - (node_count - 1)/2
    alpha_vector = alpha_0 + n_vector*beta

    mesh_x, mesh_alpha = np.meshgrid(x, alpha_vector, indexing='ij')

    mw = mass * omega
    N = (mw  / np.pi)**0.25
    sigma = np.sqrt(2/mw)
    sigma_2 = 2/mw
    coherent_xn = N * np.exp(-(mesh_x - sigma*mesh_alpha)**2/sigma_2)

    # Take outer product of that vector with itself
    coherent_xnm = np.repeat(coherent_xn[:,:,np.newaxis], node_count, axis=2)
    coherent_xnm = np.multiply(coherent_xnm,
                               np.swapaxes(coherent_xnm,1,2).conj())

    # Do tensor double contraction (np.tensordot(a,b,2)) of that matrix with
    # density matrix

    up_state_stack = np.repeat(up_state[np.newaxis,:,:], resolution, axis=0)
    down_state_stack = np.repeat(down_state[np.newaxis,:,:], resolution, axis=0)

    P_x = (p_up*np.einsum('nij,nji->n', coherent_xnm, up_state_stack)
            + p_down*np.einsum('nij,nji->n', coherent_xnm, down_state_stack))

    return P_x, x
