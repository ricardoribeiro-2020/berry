"""
 This program reads a set of wavefunctions for different k and bands and translates that
 to another set for different points in real space,
 the functions become a function of k instead of r
"""

import sys
import time
import numpy as np
from findiff import Gradient

import joblib

# These are the subroutines and functions
import contatempo
from headerfooter import header, footer
import loaddata as d

###################################################################################
if __name__ == "__main__":
    header("R2K", str(d.version), time.asctime())

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

    # Reading data needed for the run

    WFCDIRECTORY = str(d.wfcdirectory)
    print("     Directory where the wfc are:", WFCDIRECTORY)
    NKX = d.nkx
    NKY = d.nky
    NKZ = d.nkz
    print("     Number of k-points in each direction:", NKX, NKY, NKZ)
    NKS = d.nks
    print("     Total number of k-points:", NKS)

    NR = d.nr
    print("     Total number of points in real space:", NR)
    NPR = d.npr
    print("     Number of processors to use", NPR)

    NBND = d.nbnd
    print("     Number of bands:", NBND)

    DK = float(d.step)  # Defines the step for gradient calculation DK
    print("     k-points step, DK", DK)
    print()

    KPOINTS = d.kpoints
    print("     kpoints loaded")  # KPOINTS = np.zeros((NKS,3), dtype=float)

    R = d.r
    print("     rpoints loaded")  # R = np.zeros((NR,3), dtype=float)

    occupations = d.occupations
    print("     Occupations loaded")  # occupations = np.array(occupat)

    eigenvalues = d.eigenvalues
    print("     Eigenvalues loaded")  # eigenvalues = np.array(eigenval)

    PHASE = d.phase
    print("     Phases loaded")  # PHASE = np.zeros((NR,NKS),dtype=complex)

    with open("ijltonk.npy", "rb") as fich:
        ijltonk = np.load(fich)  # ijltonk converts kx,ky,kz to nk
    fich.close()
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

    grad = Gradient(h=[DK, DK], acc=2)  # Defines gradient function in 2D
    ##################################################

    BANDSR = {}  # Dictionary with wfc for all points and bands
    BANDSG = {}  # Dictionary with wfc gradients for all points and bands

    for banda in range(NBNDMIN, NBNDMAX):  # For each band
        wfct_k = np.zeros((NR, NKS), dtype=complex)
        for kp in range(NKS):
            if (
                signalfinal[kp, banda] == -1
            ):  # if its a signaled wfc, choose interpolated
                infile = (
                    WFCDIRECTORY
                    + "/k0"
                    + str(kp)
                    + "b0"
                    + str(bandsfinal[kp, banda])
                    + ".wfc1"
                )
            else:  # else choose original
                infile = (
                    WFCDIRECTORY
                    + "/k0"
                    + str(kp)
                    + "b0"
                    + str(bandsfinal[kp, banda])
                    + ".wfc"
                )
            wfct_k[:, kp] = np.load(infile)
        print(
            "     Finished reading wfcs of band ",
            str(banda),
            "from files.   {:5.2f}".format((time.time() - STARTTIME) / 60.0),
            " min",
        )
        sys.stdout.flush()

        wfcpos = {}  # Dictionary with wfc for all points
        wfcgra = {}  # Dictionary with wfc gradients for all points

        wfcpos = {
            posi: PHASE[posi, ijltonk[:, :, 0]] * wfct_k[posi, ijltonk[:, :, 0]]
            for posi in range(NR)
        }
        wfcgra = {posi: grad(wfcpos[posi]) for posi in range(NR)}
        #    for posi in range(NR):
        #      wfcpos[posi] = PHASE[posi,ijltonk[:,:,0]]*wfct_k[posi,ijltonk[:,:,0]]

        #      wfcgra[posi] = grad(wfcpos[posi])                  # Complex gradient

        BANDSR[banda] = wfcpos  # add to dictionary
        BANDSG[banda] = wfcgra
        print(
            "     Finished band ",
            str(banda),
            "   {:5.2f}".format((time.time() - STARTTIME) / 60.0),
            " min",
        )
        sys.stdout.flush()

        # Saving files
        joblib.dump(wfcpos, "./wfcpos" + str(banda) + ".gz", compress=3)
        joblib.dump(wfcgra, "./wfcgra" + str(banda) + ".gz", compress=3)

    sys.stdout.flush()

    # sys.exit("Stop")

    ###################################################################################
    # Finished
    footer(contatempo.tempo(STARTTIME, time.time()))
