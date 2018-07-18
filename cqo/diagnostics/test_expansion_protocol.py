import numpy as np
from cqo.units import hbar
from cqo.simulation import expansion_protocol, CoherentState
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Test expansion_protocol by inputing a) a coherent state b) a momentum eigenstate

def gaussian_wavefn(x, t, m, a):

    sigma = a + 1j*hbar*t/m

    N = (a/sigma)**1.5

    return N * np.exp(-x**2 / 2 / sigma)

res = 64

m = 1e-22
w = 1e5
t = 2 / w

extent = 3 + t * w

state = CoherentState(-1+1j, m*w)

sigma = np.sqrt(hbar/m/w)

coords = np.linspace(-extent*sigma, extent*sigma, num=res)

rho = [[state.sample(x, x_) for x_ in coords] for x in coords]
rho = np.array(rho)

rho_t = expansion_protocol(rho, coords, m, w, t, debug=False)

x, y = np.meshgrid(coords, coords)

a = hbar/m/w
rho_analytic = gaussian_wavefn(x, 0, m, a) * gaussian_wavefn(y, 0, m, a).conj()
rho_t_analytic = gaussian_wavefn(x, t, m, a) * gaussian_wavefn(y, t, m, a).conj()

def plot(x, y, state, title):

    fig0 = plt.figure()
    #ax = fig0.gca(projection = '3d')
    #surf = ax.plot_surface(x, y, state)
    plt.pcolor(x,y,np.abs(state))
    plt.colorbar()
    plt.title(title)

plot(x, y, rho, "Initial state")

plot(x, y, rho_analytic, "Initial state (Analytic)")

plot(x, y, rho_t, "Free flight")

plot(x, y, rho_t_analytic, "Free flight (Analytic)")

plt.show()

