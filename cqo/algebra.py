import numpy as np


# Hermitian Conjugate
def H(A):
    return A.conj().T

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

        dim_B = int(len(O) / dim_A)

        O_swapped = np.zeros(size(O))

        # Sort columns
        for col in range(len(O)):

            # Find A index
            A_idx = np.floor((col)/dim_B)
            # Find B index
            B_idx = np.mod(col, dim_B)

            new_col = A_idx + (dim_A * B_idx)

            O_swapped[:, new_col] = O[:, col]

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
        n/mxn/m ndarray: O conditioned on subsystem_vector
    """

    # Validate n mod m = 0
    if np.mod(len(O), len(subsystem_vector)) != 0:
        raise MathError(('{0}x{0} operator cannot be conditioned by '
                         '{1}D vector').format(len(O), len(subsystem_vector)))

    dim_A = int(len(O) / len(subsystem_vector))

    condition_matrix = np.kron(subsystem_vector, np.eye(dim_A))

    O_A = condition_matrix @ O @ H(condition_matrix)

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
        raise MathError(('No valid decomposition of {0}D Hilbert space by '
                        ' {1}D subspace.').format(len(O), dim_A))

    # Validate dim_A != 0
    if dim_A == 0:
        raise MathError('Partial trace undefined for |A| = 0')

    dim_B = int(len(O) / dim_A)

    I_B = np.eye(dim_B)

    # Initialise reduced operator
    O_A = np.zeros((dim_A, dim_A), dtype=np.complex128)

    for i in range(dim_B):
        projector_i = np.zeros(dim_B)
        projector_i[i] = 1


        O_A += condition_subsystem(O, projector_i)

    return O_A


