"""
 This program calculates the Berry connections
"""
from multiprocessing import Pool, Array

import sys
import time
import ctypes

import numpy as np
import joblib

# This are the subroutines and functions
from contatempo import tempo, inter_time
from headerfooter import header, footer
import loaddata as d

# pylint: disable=C0103
###################################################################################
def berry_connect(band0, band1):
    """Calculates the Berry connection."""

    ### Reading data
    wfcpos = np.load("./wfcpos" + str(band1) + ".npy", mmap_mode="r")

    # Calculation of the Berry connection
    berry_connection = np.zeros(wfcgra[0].shape, dtype=complex)

    for posi in range(d.nr):
        berry_connection += 1j * wfcpos[posi].conj() * wfcgra[posi]
    ##  we are assuming that normalization is \sum |\psi|^2 = 1
    ##  if not, needs division by d.nr
    berry_connection /= d.nr

    print("          Finished connection to band ", str(band1), "   ", inter_time(time.time() - STARTTIME))
    sys.stdout.flush()

    # output units of Berry connection are bohr
    filename = "./berryCon" + str(band1) + "_" + str(band0) + ".npy"
    np.save(filename, berry_connection)


###################################################################################
if __name__ == "__main__":
    header("BERRY CONNECTION", d.version, time.asctime())

    STARTTIME = time.time()  # Starts counting time
    GRA_SIZE = d.nr * 2 * d.nkx * d.nky
    GRA_SHAPE = (d.nr, 2, d.nkx, d.nky)

    if len(sys.argv) == 2:
        NUM_BANDS = int(sys.argv[1])
        print(
            "     Will calculate all combinations of bands from 0 up to "
            + str(NUM_BANDS)
        )
        for bandwfc in range(NUM_BANDS + 1):
            gra_base = Array(ctypes.c_double, 2 * GRA_SIZE, lock=False)
            wfcgra = np.frombuffer(gra_base, dtype=complex).reshape(GRA_SHAPE)
            wfcgra[:] = np.load("./wfcgra" + str(bandwfc) + ".npy")[:]


            print("     Calculating Berry connection for band " + str(bandwfc)); 
            sys.stdout.flush()
            with Pool(NUM_BANDS + 1) as pool:
                pool.starmap(berry_connect, [(bandwfc, band) for band in range(NUM_BANDS + 1)])

    elif len(sys.argv) == 3:
        print(
            "     Will calculate just for band "
            + str(sys.argv[1])
            + " and "
            + str(sys.argv[2])
        )
        bandwfc = int(sys.argv[1])
        gradwfc = int(sys.argv[2])

        wfcpos = np.load("./wfcpos" + str(bandwfc) + ".npy", mmap_mode="r")
        wfcgra = np.load("./wfcgra" + str(gradwfc) + ".npy", mmap_mode="r")

        berry_connection = np.zeros(wfcgra[0].shape, dtype=complex)

        for posi in range(d.nr):
            berry_connection += 1j * wfcpos[posi].conj() * wfcgra[posi]

        berry_connection /= d.nr

        filename = "./berryCon" + str(bandwfc) + "_" + str(gradwfc) + ".npy"
        np.save(filename, berry_connection)

    else:
        print("     ERROR in number of arguments. Has to be one or two integers.")
        print("     Stoping.")
        print()

    ##################################################################################
    # Finished
    footer(tempo(STARTTIME, time.time()))
