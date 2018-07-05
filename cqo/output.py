from matplotlib import pyplot as plt
import numpy as np
from algebra import condition_subsystem


def output(x, P_x):

    #! Create folder with time & date and key params

    #! Output graphical representation of final state

    plt.figure()
    plt.plot(x, P_x)
    plt.show()
    #plt.savefig("/home/matthewf/spatial_pdf.png")
