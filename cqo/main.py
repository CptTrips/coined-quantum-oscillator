import numpy as np
from simulation import final_state, CoherentState, SpinState, ThermalState, final_state_recursive
from algebra import projector
import units
import output



def main():

    #parser = argparse.ArgumentParser()
    """

    parser.add_argument('--opt', dest='opt', required=False)

    agrs = parser.parse_args()

    opt = args.opt
    """

    # OPTIONS !!! PUT THESE IN A CFG/TURN THESE INTO ARGUMENTS !!!

    N = 1

    hbar = units.hbar


    # Trap parametrs

    #mass = 3.51e3 * 4/3 * np.pi * (5e-8)**3
    #omega = 2 * np.pi * 1.5e5
    mass = 1e-6

    omega = 2 * np.pi * 1.5e5

    spins = 1 # Defect denisties of 2e24 m^-3 reported in Kern et al PRB 2017

    # Interaction strength. Appears in Hamiltonian as hbar*l*S_z*x
    # (See Scala et al PRL 2013)
    l = (0.015 * omega * spins) * np.sqrt(2 * mass * omega / hbar)

    alpha = 2 * (2 / (mass * omega**2)) * hbar * l

    alpha_0 = 0


    # Decoherence paramters

    Gamma_sc = 1e4 # scattering events s^-1. See Romero-Isart PRA 2011 eq. 10

    # off-diagonals decay as exp(-gamma*T*(x-x`)^2)
    gamma = Gamma_sc * (2 * mass * omega / hbar)

    T = omega / (4 * np.pi)

    gamma_T = gamma * T


    # Thermal parameters

    occupancy = 65 # From Photon Recoil paper

    beta = np.log((1/occupancy) + 1)/omega


    # Simulation paramters

    resolution = 64

    error = 5e-4


    # Coin operators

    Haddamard = (1/np.sqrt(2))*np.array([
        [1, 1],
        [1, -1]
    ])

    balanced_flip =(1/np.sqrt(2))*np.array([
        [1, 1j],
        [1j, 1]
    ])

    identity = np.eye(2)


    # Initialisation

    COIN_OP = balanced_flip

    walk_state = CoherentState(alpha_0, mass*omega)
    #walk_state = ThermalState(beta, omega, mass)

    spin_state = SpinState(projector(2,0))

    sample_points = np.linspace(-alpha, (N+1)*alpha, (N+3)*resolution)

    method = final_state


    # Simulation

    final_pdf = [(method(N, x, x, 0, 0, walk_state,
                              spin_state, COIN_OP, alpha, gamma_T, error)
                  + method(N, x, x, 1, 1, walk_state,
                                spin_state, COIN_OP, alpha, gamma_T, error))
                 for x in sample_points]

    # Output

    output.output(sample_points, final_pdf)


# Do not run if imported
if __name__ == "__main__":
    main()
