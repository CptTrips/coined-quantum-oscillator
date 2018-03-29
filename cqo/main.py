import numpy as np
import matplotlib.pyplot as plt

# OPTIONS !!! PUT THESE IN A CFG/TURN THESE INTO ARGUMENTS !!!
N = 2
LAMBDA = 0.1
SIM_DURATION = 50
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
down_projector = eye(2) - up_projector

spin_state = up_projector

particle_state = zeros(accessed_state_count, accessed_state_count)
particle_state[0][0] = 1

state = kron(spin_state, particle_state)

coin_op = kron(COIN_OP, identity(accessed_state_count))

translate_op = zeros(accessed_state_count)
for i in range(accessed_state_count):
    translate_op[i][i-1] = 1

shift_op = kron(up_projector, eye(accessed_state_count))
            + kron(down_projector, translate_op)

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
plt.savefig('C:\Users\0912 esmé\PhD\programming\coined-quantum-oscillator\output')