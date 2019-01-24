from cqo.simulation import CoherentState
import numpy as np
from matplotlib import pyplot as plt
from cqo.units import hbar

N = 3

norm = np.zeros((N,))

mw_0 = 1e6*1e-18

h = hbar

for i in range(N):

    mw = mw_0 * 100**i

    state = CoherentState(0, mw)

    sigma = np.sqrt(h / (2 * mw))

    sample_points = np.linspace(-4*sigma, 4*sigma)

    P_x = [state.sample(x,x) for x in sample_points]

    width = sample_points[1] - sample_points[0]

    norm[i] = sum(P_x)*width

    plt.figure()

    plt.plot(sample_points, P_x)

plt.figure()

plt.plot(norm)

plt.axis([0, N-1, 0, 1.1])

plt.show()
