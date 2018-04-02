import numpy as np
import matplotlib.pyplot as plt

# OPTIONS !!! PUT THESE IN A CFG/TURN THESE INTO ARGUMENTS !!!
N = 1
LAMBDA = 0.1
SIM_DURATION = 200
COIN_OP = (1/np.sqrt(2))*np.array([
    [1, 1],
    [1, -1]
])

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

        dim_B = len(O) / dim_A

        O_swapped = np.zeros(size(O))

        # Sort columns
        for col in range(len(O)):

            # Find A index
            A_idx = np.floor((col)/dim_B)
            # Find B index
            B_idx = mod(col, dim_B)

            new_col = A_idx + (dim_A * B_idx)

            O_swapped[:, new_col] = O[:, col]

    dim_B = len(O) / dim_A

    O_cols = swap_cols(O, dim_A)

    O_swapped = swap_cols(O_cols.T(), dim_A)

    O_swapped = O_swapped.T()




def partial_trace(O, dim_A):
    """Partial trace of an operator

    Takes an operator which lives in a Hilbert space C = A \otimes B and 
    computes its projection into just A.
    
    Args:
        O (nxn ndarray): Operator to take the partial trace of
        dim_A (int): Dimension of the remaining (first) Hilbert space.

    Returns:
        dim_Axdim_A ndarray: Partial trace of O over system B.
    """

    # Validate dim(O) mod dim_A = 0
    if len(O) mod dim_A != 0:
        raise MathError(('No valid decomposition of {0}D Hilbert space by '
                        ' {1}D subspace.').format(len(O), dim_A))

    # Validate dim_A != 0
    if dim_A == 0:
        raise MathError('Partial trace undefined for |A| = 0')

    dim_B = len(O) / dim_A

    I_B = eye(dim_B)

    O_A = zeros(dim_A)

    for i in range(dim_B):
        projector_i = np.zeros((dim_B,1))
        projector_i[i] = 1

        conditional_vector = np.kron(projector_i, I_B)

        O_A += conditional_vector.T() @ O @ conditional_vector

    return O_A

# Prepare basis & initial state
# Number of position states the particle will access
accessed_state_count = N*SIM_DURATION + 1

# Initial spin state
spin_up_projector = [
    [1, 0],
    [0, 0]
]
spin_down_projector = np.eye(2) - spin_up_projector

spin_state = spin_up_projector

# Initial particle state
particle_vector_state = np.array([1] + [0] * (accessed_state_count - 1))
particle_state = np.outer(particle_vector_state, particle_vector_state)

state = np.kron(spin_state, particle_state)

# Full coin operator
coin_op = np.kron(COIN_OP, np.eye(accessed_state_count))

# Translation operator
shift_op = np.zeros((accessed_state_count, accessed_state_count))
for i in range(accessed_state_count):
    shift_op[i][i-1] = 1

# Controlled-translation operator
up_projector = np.kron(spin_up_projector, np.eye(accessed_state_count))
cshift_op =  up_projector + np.kron(spin_down_projector, shift_op)

# Iterate over walk steps
for i in range(N*SIM_DURATION):
    #import pdb; pdb.set_trace()
#   - apply coin
    state = coin_op @  state @  H(coin_op)

#   - apply shift
    state = cshift_op @ state @ H(cshift_op)

# Output statistics
#   - create new output folder
#   - save state
#   - pdf
#   - entanglement

plt.plot(np.diag(state))
plt.savefig("/home/matthewf/output_pdf.png")
