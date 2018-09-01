import logging
import argparse
import numpy as np
from cqo.simulation import (final_state,
                        CoherentState,
                        SpinState,
                        ThermalState,
                        final_state_recursive,
                        expansion_protocol,
                        ThermalGaussian,
                        CoherentGaussian,
                        MultiGaussianWalk,
                        ClassicalMGW)
from cqo.algebra import projector
import cqo.units as units
import cqo.output as output


def create_mesh(N, alpha, width, resolution):

    sample_min = -3*width

    sample_max = N*alpha + 3*width

    sample_points = np.linspace(sample_min, sample_max, (N+1)*resolution)

    mesh_x, mesh_x_ = np.meshgrid(sample_points, sample_points)

    mesh = np.array([mesh_x, mesh_x_]).transpose(1, 2, 0)

    return mesh, sample_points

def main(resolution = 7, run_expansion=False):

    logging.basicConfig(level = logging.DEBUG)

    ### Parameters ###

    N = 5 # Walk steps

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
    omega = 2 * np.pi * 1.5e5

    tscale = (2*np.pi)/omega

    lscale = np.sqrt(hbar / (2 * mass * omega))

    """
    Magnetic field gradient: 5e2 T m^-1
    (See Scala et al PRL 2013)
    """
    g_NV = 2
    mu_B = 9.27e-24
    dB_dz = 5e6
    F = g_NV * mu_B * dB_dz

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
    occupancy = 25 # 0.5, 5, 50

    if occupancy < 1e-3:
        walk_state = CoherentState(alpha_0, mass*omega)
        quantum_walk_state = CoherentGaussian(walk_state)
    else:
        beta = np.log((1/occupancy) + 1)/omega/hbar
        walk_state = ThermalState(beta, omega, mass)
        quantum_walk_state = ThermalGaussian(walk_state)

    # Free flight time

    t_free = 9e0 / omega

    # Simulation paramters

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

    mesh, sample_points = create_mesh(N, alpha, walk_state.width, resolution)

    # Calculation method

    method = final_state

    ### Simulation ###

    # Quantum Walk

    quantum_state = [[[quantum_walk_state], []],
                 [[], []]]

    quantum_walk = MultiGaussianWalk(quantum_state, alpha, COIN_OP, gamma_T)

    quantum_walk.step(N, error=0)

    logging.info("Sampling {} points".format(mesh.shape[0:2]))

    rho_walk = np.array([
        [
            [
                [
                    quantum_walk.sample(x_vec[0], x_vec[1], s, s_) for s_ in [0,1]
                ] for s in [0,1]
            ] for x_vec in mesh_row
        ] for mesh_row in mesh])

    classical_walk = ClassicalMGW(quantum_walk_state, alpha)

    classical_walk.step(N)

    classical_pdf = np.array(
        [classical_walk.sample(x) for x in sample_points]
    )

    if run_expansion:
        # Evolve under free-flight

        quantum_walk.quarter_period(mass, omega)

        # Get edge values of mesh

        x_min = mesh.min()

        x_max = mesh.max()

        # Add max and min velocities (from mgw)

        momentum_list = np.array(quantum_walk.list_b())

        v_min = momentum_list[:,0].min() / mass

        v_max = momentum_list[:,0].max() / mass

        quantum_walk.free_flight(mass, t_free)

        x_min += v_min * t_free

        x_max += v_max * t_free

        # Generate final mesh

        resolution_final = mesh.shape[0]

        coords_final = np.linspace(x_min, x_max, resolution_final)

        mesh_x_final, mesh_x__final = np.meshgrid(coords_final, coords_final)

        mesh_final = np.array([mesh_x_final, mesh_x__final]).transpose(1,2,0)

        rho_final_0_0 = np.array([
            [
                quantum_walk.sample(x_vec[0], x_vec[1], 0, 0) for x_vec in mesh_row
            ] for mesh_row in mesh_final
        ])


        rho_final_1_1 = np.array([
            [
                quantum_walk.sample(x_vec[0], x_vec[1], 1, 1) for x_vec in mesh_row
            ] for mesh_row in mesh_final
        ])

        #import pdb; pdb.set_trace()


    ### Output ###

    walk_0 = np.diag(rho_walk[:,:,0,0])
    walk_1 = np.diag(rho_walk[:,:,1,1])

    rho_walk_dm = np.block([
        [rho_walk[:,:,0,0], rho_walk[:,:,0,1]],
        [rho_walk[:,:,1,0], rho_walk[:,:,1,1]]])

    if run_expansion:

        final_0 = np.diag(rho_final_0_0)
        final_1 = np.diag(rho_final_1_1)

        print("Displacement: {}\nWidth: {}".format(alpha, walk_state.width))

        x_final = coords_final

    output.draw_walk(sample_points, walk_0, walk_1, classical_pdf, "Walk PDF",
                      show=False)

    output.draw_density_matrix(rho_walk_dm, "Walk Density Matrix",
                               show=(not run_expansion))

    if run_expansion:

        output.draw_pdf(x_final, final_0, final_1, "Post-expansion PDF",
                          show=False)

        output.draw_expansion(sample_points, walk_0, walk_1,
                              x_final, final_0, final_1)


# Do not run if imported
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--res', dest='resolution', type=int)

    args = vars(parser.parse_args())

    null_args = []

    for k, v in args.items():
        if not v:
            null_args.append(k)

    for k in null_args:
        args.pop(k)

    if len(args) > 0:
        main(**args)
    else:
        main()

