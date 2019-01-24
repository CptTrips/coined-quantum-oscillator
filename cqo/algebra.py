import numpy as np
from itertools import combinations
import math


def H(A):
    """Outputs the Hermitian conjugate of A"""
    return A.conj().T

def projector(dim, i):
    """Builds the projection operator on to the ith eigenvector of a dimension
    dim operator.

    Args:
        dim (int): Dimension of the Hilbert space
        i (int): Eigenvector to project on to

    Returns:
        dimxdim ndarray: Projector onto ith eignenvector

    """

    # Validate dim, i positive
    if i < 0:
        raise ValueError("Eigenvector label must be positive")

    if dim < 1:
        raise ValueError("Hilbert space dimension must be > 0")

    # Validate i < dim
    if i >= dim:
        raise ValueError("Eigenvector label must be less than dimension - 1")

    # ith eigenvector
    eigenvector = np.zeros(dim, dtype=np.complex128)
    eigenvector[i] = 1

    projector = np.outer(eigenvector, eigenvector)

    return projector

def tensor_swap(O, dim_A):
    """Swaps kron(A,B) with kron(B,A)

    Args:
        O (nxn ndarray): Input operator
        dim_A (int): Dimension of the first subsystem

    Returns:
        nxn ndarray
    """

    def swap_cols(O, dim_A):
        """ Swaps columns from A major to B major
        """

        if np.mod(dim_A, 1) != 0:
            raise ValueError("Dimension of A must be an integer")


        dim_B = int(len(O) / dim_A)

        O_swapped = np.zeros(O.shape)

        # Sort columns
        for col in range(len(O)):

            # Find A index
            A_idx = int(np.floor((col)/dim_B))
            # Find B index
            B_idx = np.mod(col, dim_B)

            new_col = A_idx + (dim_A * B_idx)

            O_swapped[:, new_col] = O[:, col]

        return O_swapped

    if np.mod(len(O), dim_A) != 0:
         raise ValueError(("Dimension {0} operator cannot be factored by "
                           "dimension {1} subsystem").format(len(O), dim_A))

    if (len(O) == dim_A):
        raise ValueError("Cannot swap when subsystem dimension == full operator dimension")

    if len(O.shape) != 2:
        raise ValueError("Can only swap rank 2 matrices")

    if O.shape[0] != O.shape[1]:
        raise ValueError("Can only swap square matrices")

    dim_B = int(len(O) / dim_A)

    O_cols = swap_cols(O, dim_A)

    O_swapped = swap_cols(O_cols.T, dim_A)

    O_swapped = O_swapped.T

    return O_swapped

def condition_subsystem(O, subsystem_vector):
    """Conditions on subsystem B of an operator.

    Can be used to condition on a state of a subsystem.

    Args:
        O (nxn ndarray): Operator to be conditioned
        subsystem_vector (mx1 ndarray): Vector to condition on

    Returns:
        (n/m)x(n/m) ndarray: O conditioned on subsystem_vector
    """

    # Validate n mod m = 0
    if np.mod(len(O), len(subsystem_vector)) != 0:
        raise ValueError(('{0}x{0} operator cannot be conditioned by '
                         '{1}D vector').format(len(O), len(subsystem_vector)))

    dim_A = int(len(O) / len(subsystem_vector))

    condition_matrix = np.kron(subsystem_vector, np.eye(dim_A))

    O_A = condition_matrix @ O @ H(condition_matrix)

    return O_A

def postselect(O, vector):

    # Throw warning if O not valid density matrix

    O_A = condition_subsystem(O, vector)

    # Normalise
    p = np.trace(O_A)
    if p>0:
        O_A = O_A / p
    else:
        O_A = np.zeros(O_A.shape)

    return O_A, p

def partial_trace(O, dim_A):
    """Partial trace of an operator

    Takes an operator which lives in a Hilbert space C = A otimes B and 
    computes its projection into just A.
    
    Args:
        O (nxn ndarray): Operator to take the partial trace of
        dim_A (int): Dimension of the remaining (first) Hilbert space.

    Returns:
        dim_Axdim_A ndarray: Partial trace of O over system B.
    """

    # Validate dim(O) mod dim_A = 0
    if np.mod(len(O), dim_A) != 0:
        raise ValueError(('No valid decomposition of {0}D Hilbert space by '
                        ' {1}D subspace.').format(len(O), dim_A))

    # Validate dim_A != 0
    if dim_A == 0:
        raise ValueError('Partial trace undefined for |A| = 0')

    if dim_A == len(O) or dim_A == 1:
        raise ValueError('Subsystem dimension cannot equal full system'
                         ' dimension or equal 1')

    dim_B = int(len(O) / dim_A)

    # Initialise reduced operator
    O_A = np.zeros((dim_A, dim_A), dtype=np.complex128)

    for i in range(dim_B):

        vector_i = np.zeros(dim_B)
        vector_i[i] = 1

        O_i = condition_subsystem(O, vector_i)

        O_A += O_i

    return O_A

def binary_combinations(N, S):
    """Returns all binary strings of length N whose bits sum to S
    """

    if S > N:
        raise ValueError("Sum cannot be larger than number of bits")

    if S == 0:
        return np.array([np.zeros((N,), dtype=np.int)])

    # List all the ways you can choose S numbers from 0 to N-1

    c = np.array([i for i in combinations(range(N), S)],
                 dtype=np.int) # Could use np.choose

    # Create an array of N by N choose S zeros

    bc = np.zeros((len(c), N), dtype=np.int)

    # set the array of zeroes to 1, addressed by the list of combinations

    # add a dimension to c which labels the outer index

    i = np.tile(np.arange(len(c)), (S,1)).T

    i = i.reshape((S*len(c), ))

    c = c.reshape((S*len(c), ))

    bc[i, c] = 1

    return bc

def choose(n, r):

    return math.factorial(n) / math.factorial(n - r) / math.factorial(r)

def binomial_distribution(n):

    choose_n = [choose(n, r) for r in range(n+1)]

    norm = sum(choose_n)

    return [ncr / norm for ncr in choose_n]
