from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from itertools import product
from cqo.simulation import ThermalState, CoherentState
from cqo.units import hbar
import numpy as np

# Display a thermal density matrix and show it is normalised to 1

output_folder = '/home/matthewf/PhD/programming/coined-quantum-oscillator/output'

def beta(n, omega):

    return np.log(1 + 1/n) / (omega*hbar)

mass = 1#3.51e3 * 4/3 * np.pi * (5e-8)**3
omega = 1#2 * np.pi * 1.5e5
n = 0.001

rho = ThermalState(beta(n, omega), omega, mass)

limit = 3*rho.width

N = 3

base_res = 32

fidelity = np.zeros((N,))
normalisation = np.zeros((N,))

for i in range(1,N):

    res = base_res * 2**i

    x_array = np.linspace(-limit, limit, res)

    width = x_array[1] - x_array[0]

    fidelity[i] = 2**i

    pdf = [rho.sample(x,x) for x in x_array]

    normalisation[i] = width*sum(pdf)

    density_matrix = np.array([[rho.sample(x, x_) for x in x_array] for x_ in x_array])

    X, Y = np.meshgrid(x_array, x_array)

    fig = plt.figure()

    plt.subplot(2,1,1)
    plt.pcolor(X, Y, np.abs(density_matrix))
    plt.colorbar()

    plt.subplot(2,1,2)
    plt.plot(np.diag(X), np.diag(density_matrix))

    plt.title('\\rho(x,x\')')


fig2 = plt.figure()
plt.plot(fidelity, normalisation)
plt.ylim(ymin=0)
plt.title('Norm vs resolution');

# Normalisation vs temperature
def norm(thermal_state):

    limit = 5 * thermal_state.width

    x_array = np.linspace(-limit, limit, 16)

    pdf = [thermal_state.sample(x,x) for x in x_array]

    return np.trapz(pdf, x_array)

n_array = 10**np.arange(-6, 1.5, 0.25)

beta_array = beta(n_array, omega)

norm_array = [norm(ThermalState(b, omega, mass)) for b in beta_array]

plt.figure()
plt.subplot(2,1,1)
plt.plot(1/beta_array, norm_array, label='Norm')
plt.plot([0, 1/beta_array[-1]], [1, 1], '--')
plt.ylim(ymin=0, ymax=1.1)
plt.xlim(xmin=0)
plt.title('Norm vs Temperature')
plt.xlabel('1/beta')
plt.legend()

# Width vs temperature (classical high temperature asymptotics)

width_array = [ThermalState(b, omega, mass).width for b in beta_array]

c_width_array = [ThermalState(b, omega, mass).coherence_width for b in beta_array]

high_t_width = np.sqrt(1 / (mass * omega**2 * beta_array))

gs_width = np.sqrt(hbar/(2*mass*omega))

plt.subplot(2,2,3)
plt.plot(n_array, width_array, label='QHO width')
plt.plot(n_array, high_t_width, label='CHO width')
plt.plot(n_array, c_width_array, label='Coherence length')
plt.plot([0, n_array[-1]], [gs_width]*2, '--', label='Quantum width')
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.title('Width vs Temperature')
plt.xlabel('<n>')
plt.legend()

plt.subplot(2,2,4)
plt.plot(1/beta_array, width_array/high_t_width)
plt.plot([0,1/beta_array[-1]], [1,1], '--')
plt.xlim(xmin=0)
plt.title('Classical width / Quantum Width')
plt.xlabel('1/beta')

# Comparison of low-T state with ground state

b = 1e35

low_t_state = ThermalState(b, omega, mass)

x_array = np.linspace(-3*low_t_state.width, 3*low_t_state.width)

gs_pdf = [CoherentState(0, mass * omega).sample(x, x) for x in x_array]

low_t_pdf = [low_t_state.sample(x, x) for x in x_array]

plt.figure()
plt.plot(x_array, gs_pdf, '--', label='Ground State')
plt.plot(x_array, low_t_pdf, label='Low T State')
plt.plot([low_t_state.width]*2, [0, max(low_t_pdf)], '--')
plt.legend()
plt.title('Low T State vs Ground State')

# Attribute width vs calculated width

def calculate_width(t):

    exponent = np.log(t.sample(0,0) / t.sample(t.width, t.width))

    return t.width * np.sqrt( 1 / (2 * exponent))

thermal_state_array = [ThermalState(b, omega, mass) for b in beta_array]

calculated_width_array = [calculate_width(t) for t in thermal_state_array]

plt.figure()
plt.plot(1/beta_array, calculated_width_array, label='Calculated width')
plt.plot(1/beta_array, width_array, '--', 'Width attribute')
plt.legend()
plt.xlabel('1/beta')


# Width vs theoretical value (Quantum Optics in Phase Space)

def theoretical_width(beta, omega, m):

    return np.sqrt(hbar / (2 * m * omega) / np.tanh(beta * hbar * omega/2))

theoretical_width_array = [theoretical_width(b, omega, mass) for b in beta_array]

plt.figure()

plt.subplot(2,1,1)
plt.plot(beta_array, width_array, label='State width')
plt.plot(beta_array, theoretical_width_array, label='Theoretical width')
plt.legend()

plt.subplot(2,1,2)
plt.plot(beta_array, np.array(width_array)/np.array(theoretical_width_array))

plt.show()
