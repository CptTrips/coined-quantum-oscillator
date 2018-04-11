import numpy as np
import simulate
import output



def main():

    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--opt', dest='opt', required=False)

    agrs = parser.parse_args()

    opt = args.opt
    """

    # OPTIONS !!! PUT THESE IN A CFG/TURN THESE INTO ARGUMENTS !!!

    N = 2
    SIM_DURATION = 20

    mass = 1
    omega = 1
    Lambda = 0.05
    alpha_0 = 0
    gamma = 5

    resolution = 20*SIM_DURATION
    error = 0.01

    Haddamard = (1/np.sqrt(2))*np.array([
        [1, 1],
        [1, -1]
    ])

    balanced_flip =(1/np.sqrt(2))*np.array([
        [1, 1j],
        [1j, 1]
    ])

    identity = np.eye(2)

    COIN_OP = balanced_flip

    # Calculate state after quantum walk
    final_state = simulate.simulate(COIN_OP, SIM_DURATION, gamma)

    # Calculate spatial pdf from final state
    P_x, x = simulate.spatial_pdf(final_state, mass, omega, alpha_0, Lambda, resolution, error)

    # Output
    output.output(final_state, P_x)


# Do not run if imported
if __name__ == "__main__":
    main()
