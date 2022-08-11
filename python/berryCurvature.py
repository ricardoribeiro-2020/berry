"""
# This program calculates the Berry curvature
"""

from multiprocessing import Pool, Array

import sys
import time
import ctypes

import numpy as np

# This are the subroutines and functions
from contatempo import tempo, inter_time
from headerfooter import header, footer
import loaddata as d

# pylint: disable=C0103
###################################################################################
def berry_curv(band0, band1):
    """Calculates the Berry curvature."""

    if band0 != band1:
        wfcgra1 = np.load("./wfcgra" + str(band1) + ".npy", mmap_mode="r")
    else:
        wfcgra1 = wfcgra

    berry_curvature = np.zeros(wfcgra[0].shape, dtype=complex)

    for posi in range(d.nr):
        berry_curvature += (
            1j * wfcgra[posi][0] * wfcgra1[posi][1].conj()
            - 1j * wfcgra[posi][1] * wfcgra1[posi][0].conj()
        )
    ##  we are assuming that normalization is \sum |\psi|^2 = 1
    ##  if not, needs division by d.nr
    berry_curvature /= d.nr
    print("          Finished curvature for index " + str(band1), inter_time(time.time() - STARTTIME)); sys.stdout.flush()

    filename = "./berry_curvature" + str(band0) + "-" + str(band1) + ".npy"

    np.save(filename, berry_curvature)

###################################################################################
if __name__ == "__main__":
    header("BERRY CURVATURE", d.version, time.asctime())

    STARTTIME = time.time()  # Starts counting time
    NUM_BANDS = int(sys.argv[1])
    GRA_SIZE = d.nr * 2 * d.nkx * d.nky
    GRA_SHAPE = (d.nr, 2, d.nkx, d.nky)

    if len(sys.argv) == 2:
        print(
            "     Will calculate all combinations of bands from 0 up to "
            + str(NUM_BANDS)
        )
        for gradwfc0 in range(NUM_BANDS + 1):
            gra_base = Array(ctypes.c_double, 2 * GRA_SIZE, lock=False)
            wfcgra = np.frombuffer(gra_base, dtype=complex).reshape(GRA_SHAPE)
            wfcgra = np.load("./wfcgra" + str(gradwfc0) + ".npy")

            print("     Calculating Berry curvature for band ", str(gradwfc0)); sys.stdout.flush()
            with Pool(NUM_BANDS) as pool:
                pool.starmap(berry_curv, [(gradwfc0, gradwfc1) for gradwfc1 in range(NUM_BANDS + 1)])

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
    footer(tempo(STARTTIME, time.time()))