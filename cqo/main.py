import numpy as np
import matplotlib.pyplot as plt

# OPTIONS !!! PUT THESE IN A CFG/TURN THESE INTO ARGUMENTS !!!
N = 2
LAMBDA = 0.1
SIM_DURATION = 5
COIN_OP = [
    [1, 1],
    [1, -1]
]

def H(A):
    return A.conj().T

# Prepare basis & initial state
# Number of position states the particle will access
accessed_state_count = N*SIM_DURATION + 1

up_projector = [
    [1, 0],
    [0, 0]
]
down_projector = np.eye(2) - up_projector

spin_state = up_projector

particle_state = np.zeros(accessed_state_count, accessed_state_count)
particle_state[0][0] = 1

state = np.kron(spin_state, particle_state)

coin_op = np.kron(COIN_OP, np.eye(accessed_state_count))

translate_op = np.zeros(accessed_state_count)
for i in range(accessed_state_count):
    translate_op[i][i-1] = 1

shift_op = np.kron(up_projector, np.eye(accessed_state_count)) + \
             np.kron(down_projector, translate_op)

# Iterate over walk steps
for i in range(N*SIM_DURATION):

#   - apply coin
    state = coin_op * state * H(coin_op)

#   - apply shift
    state = shift_op * state * H(shift_op)

# Output statistics
#   - create new output folder
#   - save state
#   - pdf
#   - entanglement

plt.plot(np.diag(state))
plt.savefig("C:\\outputpdf.png")