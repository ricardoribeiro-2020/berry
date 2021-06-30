"""
 This program calculates the Berry connections
"""

import sys
import time

import numpy as np
import joblib

# This are the subroutines and functions
from contatempo import tempo, inter_time
from headerfooter import header, footer
import loaddata as d

# pylint: disable=C0103
###################################################################################
def berry_connect(bandwfc0, gradwfc0):
    """Calculates the Berry connection."""

    # Reading data needed for the run
    print()
    print(
        "     Reading files ./wfcpos"
        + str(bandwfc0)
        + ".gz and ./wfcgra"
        + str(gradwfc0)
        + ".gz"
    )
    wfcpos = joblib.load("./wfcpos" + str(bandwfc0) + ".gz")
    wfcgra = joblib.load("./wfcgra" + str(gradwfc0) + ".gz")

    print(
        "     Finished reading data ",
        str(bandwfc0),
        " and ",
        str(gradwfc0),
        "         "
        + inter_time(time.time() - STARTTIME)
    )
    sys.stdout.flush()
    #  sys.exit("Stop")

    ### Finished reading data
    # Calculation of the Berry connection
    berry_connection = np.zeros(wfcgra[0].shape, dtype=complex)

    for posi in range(d.nr):
        berry_connection += 1j * wfcpos[posi].conj() * wfcgra[posi]
    ##  we are assuming that normalization is \sum |\psi|^2 = 1
    ##  if not, needs division by d.nr
    berry_connection /= d.nr

    print(
        "     Finished calculating Berry connection for index "
        + str(bandwfc0)
        + "  "
        + str(gradwfc0)
        + "  \
         \n     Saving results to file  "
        + inter_time(time.time() - STARTTIME)
    )
    sys.stdout.flush()

    filename = "./berryCon" + str(bandwfc0) + "-" + str(gradwfc0)
    # output units of Berry connection are bohr

    joblib.dump(berry_connection, filename + ".gz", compress=3)


###################################################################################
if __name__ == "__main__":
    header("BERRY CONNECTION", d.version, time.asctime())

    STARTTIME = time.time()  # Starts counting time

    if len(sys.argv) == 2:
        print(
            "     Will calculate all combinations of bands from 0 up to "
            + str(sys.argv[1])
        )
        for bandwfc in range(int(sys.argv[1]) + 1):
            for gradwfc in range(int(sys.argv[1]) + 1):
                berry_connect(bandwfc, gradwfc)

    elif len(sys.argv) == 3:
        print(
            "     Will calculate just for band "
            + str(sys.argv[1])
            + " and "
            + str(sys.argv[2])
        )
        bandwfc = int(sys.argv[1])
        gradwfc = int(sys.argv[2])
        berry_connect(bandwfc, gradwfc)

    else:
        print("     ERROR in number of arguments. Has to be one or two integers.")
        print("     Stoping.")
        print()

    ##################################################################################r
    # Finished
    footer(tempo(STARTTIME, time.time()))
