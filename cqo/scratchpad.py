import numpy as np
from matplotlib import pyplot as plt
from cqo.units import hbar, k_B
from cqo.simulation import ThermalState, expansion_protocol

def eigenvalues_of_A():

    m = 1

    omega = 1

    t_array = np.linspace(0.75,1)

    def A(t):

        A = np.array([
            [3/2, 1j, -t, -1j*t],
            [0, 1/2, 1j*t, -t],
            [0, 0, 3/2, -1j],
            [0, 0, 0, 1/2]])
        A = A + A.T

        return A

    evals = np.array([np.linalg.eigvals(A(t)) for t in t_array])

    plt.figure()

    for i in range(4):
        plt.plot(t_array, evals[:,i])

    print(evals[-1,:])

    plt.show()

    # Find t which makes A singular

    t_min = 0.9

    t_max = 1

    min_e = 1

    eps = 1e-15

    while abs(min_e.real) > eps:

        t = (t_min + t_max) / 2

        min_e = np.linalg.eigvals(A(t)).min()

        if min_e.real > 0:
            t_min = t
        elif min_e.real < 0:
            t_max = t

        print("t_min: {}, t_max: {}, min_e: {}".format(t_min, t_max, min_e))

def betahomega(occupancy):
    return np.log((1/occupancy) + 1)

def coherence_vs_temperature():

    omega = 1.5e5

    mass = 1e-18

    occupancies = 10**np.linspace(-1, 2, 10)

    betas = betahomega(occupancies) / omega / hbar

    width_ratios = [ThermalState(beta, omega, mass).coherence_width / ThermalState(beta, omega, mass).width for beta in betas]

    plt.figure()

    plt.plot(occupancies, np.log10(width_ratios))

    plt.show()

def width_vs_occupancy():

    m = 3.5e3 * 5e2 * 1e-24

    omega = 1e5

    occupancies = np.linspace(1, 100)

    betas = betahomega(occupancies)/omega/hbar

    temps = 1 / betas / k_B

    widths = np.array([ThermalState(beta, omega, m).width for beta in betas])

    gs_width = np.sqrt(np.log(2) * hbar / m / omega)

    fig = plt.figure()

    ax1 = fig.gca()

    ax1.plot(occupancies, widths*1e10)

    ax1.set_xlabel('$\\bar{n}$')
    ax1.set_ylabel('HWHM (Å)')

    ax2 = ax1.twiny()

    ax2.set_xlabel('T ($\mu$K)')

    ax2.plot(1e6*temps, gs_width*np.ones(temps.shape)*1e10)

    plt.show()

def free_flight_coherence():

    beta = 1e17
    omega = 1e5
    m = 1e-18

    rho = ThermalState(beta, omega, mass)

    # plot density matrix

    sample_points = np.linspace(-4*rho.width, 4*rho.width)

    density_matrix = np.array(
        [
            [
                rho.sample(x, x_) for x_ in sample_points
            ]
            for x in sample_points
        ]
    )

    plt.figure()

    plt.imshow(abs(density_matrix))

    plt.title("Before")

    # plot density matrix after free flight

    mesh_x, mesh_y = np.meshgrid(sample_points, sample_points)

    mesh = np.array([mesh_x, mesh_y]).transpose(1, 2, 0)

    ff_density_matrix = expansion_protocol(density_matrix, mesh, m, omega, 1 / omega)

    plt.figure()

    plt.imshow(abs(density_matrix))



if __name__ == "__main__":
    coherence_vs_temperature()

