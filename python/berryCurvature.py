"""
# This program calculates the Berry curvature
"""

import sys
import time

import numpy as np
import joblib

# This are the subroutines and functions
import contatempo
from headerfooter import header, footer
import loaddata as d

###################################################################################
def berry_curv(gradwfc00, gradwfc11):
    """Calculates the Berry curvature."""

    # Reading data needed for the run
    NR = d.nr
    print()
    print(
        "     Reading files ./wfcpos"
        + str(gradwfc00)
        + ".gz and ./wfcgra"
        + str(gradwfc11)
        + ".gz"
    )
    wfcgra0 = joblib.load("./wfcgra" + str(gradwfc00) + ".gz")
    wfcgra1 = joblib.load("./wfcgra" + str(gradwfc11) + ".gz")

    print(
        "     Finished reading data ",
        str(gradwfc00),
        " and ",
        str(gradwfc11),
        "   {:5.2f}".format((time.time() - STARTTIME) / 60.0),
        " min",
    )
    sys.stdout.flush()
    #  sys.exit("Stop")

    ### Finished reading data
    # Calculation of the Berry curvature
    berry_curvature = np.zeros(wfcgra0[0].shape, dtype=complex)

    for posi in range(NR):
        berry_curvature += (
            1j * wfcgra0[posi][0] * wfcgra1[posi][1].conj()
            - 1j * wfcgra0[posi][1] * wfcgra1[posi][0].conj()
        )
    ##  we are assuming that normalization is \sum |\psi|^2 = 1
    ##  if not, needs division by NR
    berry_curvature /= NR

    print(
        "     Finished calculating Berry curvature for index "
        + str(gradwfc00)
        + "  "
        + str(gradwfc11)
        + ".\
           \n     Saving results to file   {:5.2f}".format(
            (time.time() - STARTTIME) / 60.0
        ),
        " min",
    )
    sys.stdout.flush()

    filename = "./berry_curvature" + str(gradwfc00) + "-" + str(gradwfc11)
    # output units of Berry curvature is none

    joblib.dump(berry_curvature, filename + ".gz", compress=3)


###################################################################################
if __name__ == "__main__":
    header("BERRY CURVATURE", str(d.version), time.asctime())

    STARTTIME = time.time()  # Starts counting time

    if len(sys.argv) == 2:
        print(
            "     Will calculate all combinations of bands from 0 up to "
            + str(sys.argv[1])
        )
        for gradwfc0 in range(int(sys.argv[1]) + 1):
            for gradwfc1 in range(int(sys.argv[1]) + 1):
                berry_curv(gradwfc0, gradwfc1)

    elif len(sys.argv) == 3:
        print(
            "     Will calculate just for band "
            + str(sys.argv[1])
            + " and "
            + str(sys.argv[2])
        )
        gradwfc0 = int(sys.argv[1])
        gradwfc1 = int(sys.argv[2])
        berry_curv(gradwfc0, gradwfc1)

    else:
        print("     ERROR in number of arguments. Has to have one or two integers.")
        print("     Stoping.")
        print()

    ##################################################################################r
    # Finished
    footer(contatempo.tempo(STARTTIME, time.time()))
