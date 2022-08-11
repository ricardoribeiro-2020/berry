"""
 This program reads a set of wavefunctions for different k and bands and translates that
 to another set for different points in real space,
 the functions become a function of k instead of r
"""
from multiprocessing import Pool

import gc
import sys
import time

from findiff import Gradient

import numpy as np

# These are the subroutines and functions
from contatempo import tempo, inter_time
from headerfooter import header, footer
import loaddata as d

# pylint: disable=C0103
###################################################################################
def apply_grad(wfcidx):
    return grad(wfcidx)


def func_name(banda):
    wfct_k = np.zeros((d.nr, d.nks), dtype=complex)
    print(
        "     Reading wfcs of band ",
        str(banda),
        "from files.   ",
        inter_time(time.time() - STARTTIME),
    )
    sys.stdout.flush()

    for kp in range(d.nks):
        if signalfinal[kp, banda] == -1:  # if its a signaled wfc, choose interpolated
            infile = (
                d.wfcdirectory
                + "/k0"
                + str(kp)
                + "b0"
                + str(bandsfinal[kp, banda])
                + ".wfc1"
            )
        else:  # else choose original
            infile = (
                d.wfcdirectory
                + "/k0"
                + str(kp)
                + "b0"
                + str(bandsfinal[kp, banda])
                + ".wfc"
            )
        wfct_k[:, kp] = np.load(infile, mmap_mode="r")
    print(
        "     Finished reading wfcs of band ",
        str(banda),
        "from files.   ",
        inter_time(time.time() - STARTTIME),
    )
    sys.stdout.flush()

    t1 = time.time()
    wfcpos = np.array(
        [
            d.phase[posi, d.ijltonk[:, :, 0]] * wfct_k[posi, d.ijltonk[:, :, 0]]
            for posi in range(d.nr)
        ]
    )
    t2 = time.time()
    wfcgra = np.array([grad(wfcpos[posi]) for posi in range(d.nr)])
    t3 = time.time()
    #    for posi in range(d.nr):
    #      wfcpos[posi] = d.phase[posi,d.ijltonk[:,:,0]]*wfct_k[posi,d.ijltonk[:,:,0]]

    #      wfcgra[posi] = grad(wfcpos[posi])                  # Complex gradient

    print("     Finished positions gradients.   ", inter_time(t2 - t1))
    print("     Finished gradients gradients.   ", inter_time(t3 - t2))
    print(
        "     Saving band ", str(banda), "        ", inter_time(time.time() - STARTTIME)
    )
    sys.stdout.flush()

    # Saving files
    np.save(f"wfcpos{banda}.npy", wfcpos)
    np.save(f"wfcgra{banda}.npy", wfcgra)

    del wfcpos, wfcgra, wfct_k
    t1 = time.time()
    gc.collect()
    t2 = time.time()
    print("     Finished garbage collection.   ", inter_time(t2 - t1))

    print(
        "     Saved band ", str(banda), "        ", inter_time(time.time() - STARTTIME)
    )
    sys.stdout.flush()


if __name__ == "__main__":
    header("R2K", d.version, time.asctime())

    STARTTIME = time.time()  # Starts counting time

    if len(sys.argv) < 2:
        print("     ERROR in number of arguments.")
        print("     Have to give the number of bands that will be considered.")
        print("     One number and calculates from 0 to that number.")
        print("     Two numbers and calculates from first to second.")
        sys.exit("Stop")
    elif len(sys.argv) == 2:
        NBNDMIN = 0
        NBNDMAX = int(sys.argv[1]) + 1  # Number of bands to be considered
        print("     Will calculate bands and their gradient for bands 0 to", NBNDMAX)
    elif len(sys.argv) == 3:
        NBNDMIN = int(sys.argv[1])  # Number of lower band to be considered
        NBNDMAX = int(sys.argv[2]) + 1  # Number of higher band to be considered
        print(
            "     Will calculate bands and their gradient for bands ",
            NBNDMIN,
            " to",
            NBNDMAX - 1,
        )
    NPR = 5  # TODO change this to an input parameter

    # Reading data needed for the run

    print("     Unique reference of run:", d.refname)
    print("     Directory where the wfc are:", d.wfcdirectory)
    print("     Number of k-points in each direction:", d.nkx, d.nky, d.nkz)
    print("     Total number of k-points:", d.nks)
    print("     Total number of points in real space:", d.nr)
    print("     Number of processors to use", NPR)
    print("     Number of bands:", d.nbnd)
    print("     k-points step, dk", d.step)  # Defines the step for gradient calculation
    print()
    print("     kpoints loaded")  # d.kpoints = np.zeros((d.nks,3), dtype=float)
    print("     rpoints loaded")  # d.r = np.zeros((d.nr,3), dtype=float)
    print("     Occupations loaded")  # d.occupations = np.array(occupat)
    print("     Eigenvalues loaded")  # d.eigenvalues = np.array(eigenval)
    print("     Phases loaded")  # d.phase = np.zeros((d.nr,d.nks),dtype=complex)

    with open("bandsfinal.npy", "rb") as fich:
        bandsfinal = np.load(fich)
    fich.close()
    print("     bandsfinal.npy loaded")
    with open("signalfinal.npy", "rb") as fich:
        signalfinal = np.load(fich)
    fich.close()
    print("     signalfinal.npy loaded")
    print()
    sys.stdout.flush()
    # sys.exit("Stop")

    ################################################## Finished reading data

    grad = Gradient(h=[d.step, d.step], acc=2)  # Defines gradient function in 2D
    ##################################################

    for banda in range(NBNDMIN, NBNDMAX):
        func_name(banda)

    # sys.exit("Stop")

    ###################################################################################
    # Finished
    footer(tempo(STARTTIME, time.time()))
