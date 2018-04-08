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
    N = 1
    LAMBDA = 0.1
    SIM_DURATION = 25
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
    final_state = simulate.simulate(COIN_OP, SIM_DURATION)

    # Calculate spatial pdf from final state

    # Output
    output.output(final_state)


# Do not run if imported
if __name__ == "__main__":
    main()
