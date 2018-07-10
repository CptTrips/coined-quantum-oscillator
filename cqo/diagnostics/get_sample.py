import numpy as np
from simulation import final_state, CoherentState, SpinState

walk_state = CoherentState(0,1)
spin_state = SpinState([[1,0],[0,0]])

coin_op = np.sqrt(0.5)*np.array([[1,1],[1,-1]])

sample = final_state(25, 0, 0, 0, 0, walk_state, spin_state, coin_op, 1, 0.1)

str(sample)
