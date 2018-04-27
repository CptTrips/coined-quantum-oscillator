import numpy as np
import algebra
from algebra import H, condition_subsystem, postselect


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


