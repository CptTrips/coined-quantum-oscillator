import numpy as np
from simulation import (final_state,
                        CoherentState,
                        SpinState,
                        ThermalState,
                        final_state_recursive,
                        expansion_protocol)
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


    ### Parameters ###

    N = 2 # Walk steps

    hbar = units.hbar


    """
    Density of diamond: 3.51e3 kg/m^3
    Nanoparticle radius: ~1e-8m
    See Pflanzer thesis eq 2.64 for trap frequency
    """
    density = 3.51e3
    radius = 5e-8

    mass = density * 4/3 * np.pi * radius**3


    """
    Trap frequencies: 0.1-1 Mhz
    """
    omega = 2 * np.pi * 0.5e5

    tscale = (2*np.pi)/omega

    lscale = np.sqrt(hbar / (2 * mass * omega))

    """
    Magnetic gradient force: ~5e-22 N
    (See Scala et al PRL 2013)
    """
    F = 1e-20
    F_enhancement = 1e2
    F = F * F_enhancement

    alpha = 2 * F / mass / omega**2

    alpha_0 = 0


    """
    Decoherence rate: 2*pi*1.1e4 /s. See Romero-Isart PRA 2011 eq. 10
    off-diagonals decay as exp(-gamma*T*(x-x`)^2)
    """
    Gamma_sc = 2 * np.pi * 1.1e2

    gamma = Gamma_sc / lscale**2

    T = tscale / 2

    gamma_T = gamma * T


    """
    Reported thermal occupancy: 65 phonons (From Photon Recoil paper)
    """
    occupancy = 3e-4 # 0.5, 5, 50

    if occupancy < 1e-3:
        walk_state = CoherentState(alpha_0, mass*omega)
    else:
        beta = np.log((1/occupancy) + 1)/omega/hbar
        walk_state = ThermalState(beta, omega, mass)

    # Free flight time

    t_free = 2**0 / omega

    # Simulation paramters

    resolution = 5

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


    ### Initialisation ###

    # Initial states

    COIN_OP = balanced_flip

    spin_state = SpinState(projector(2,0))

    # Mesh

    sample_min = -4*walk_state.width

    sample_max = N*alpha + 4*walk_state.width

    sample_points = np.linspace(sample_min, sample_max, (N+1)*resolution)

    mesh_x, mesh_x_ = np.meshgrid(sample_points, sample_points)

    mesh = np.array([mesh_x, mesh_x_]).transpose(1, 2, 0)

    # Calculation method

    method = final_state

    ### Simulation ###

    # Quantum Walk

    element = lambda x, x_, s, s_: method(N, x, x_, s, s_, walk_state,
                                          spin_state, COIN_OP, alpha, gamma_T,
                                          error)

    rho_walk = np.array([
    [
        [
            [
                element(vec[0], vec[1], s, s_) for s_ in [0,1]
            ] for s in [0,1]
        ] for vec in row
    ] for row in mesh])

    # Evolve under free-flight

    calc_rho_final = lambda s, s_: \
            expansion_protocol(rho_walk[:,:,s,s_],
                               mesh,
                               mass, omega,
                               t_free)

    rho_final_0_0, coords_final = calc_rho_final(0,0)
    rho_final_1_1, coords_final = calc_rho_final(1,1)

    ### Output ###

    walk_0 = np.diag(rho_walk[:,:,0,0])
    walk_1 = np.diag(rho_walk[:,:,1,1])

    final_0 = np.diag(rho_final_0_0)
    final_1 = np.diag(rho_final_1_1)

    print("Displacement: {}\nWidth: {}".format(alpha, walk_state.width))

    x_final = coords_final[:,0,1]

    output.draw_state(sample_points, walk_0, walk_1, "Walk PDF",
                      show=False)

    output.draw_state(x_final, final_0, final_1, "Post-expansion PDF",
                      show=False)

    output.draw_expansion(sample_points, walk_0, walk_1,
                          x_final, final_0, final_1)


# Do not run if imported
if __name__ == "__main__":
    main()
