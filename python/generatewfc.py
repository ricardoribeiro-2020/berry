"""  This program reads the wfc from DFT calculations make them coherent and saves
   in separate files

  Can accept 0, 1 or 2 arguments.
  If it has 0 arguments, it will run for all k-points and bands
  If it has 1 argument, it will run just for one k-point, specified by the argument
  If it has 2 arguments, it will run just for 1 k-point and 1 band, specified by the arguments
"""
import os
import sys
import time

# This are the subroutines and functions
import contatempo
import dft
from headerfooter import header, footer
import loaddata as d

###################################################################################
if __name__ == "__main__":
    header("GENERATEWFC", time.asctime())

    STARTTIME = time.time()  # Starts counting time

    WFCDIRECTORY = str(d.wfcdirectory)
    DFTDIRECTORY = str(d.dftdirectory)

    # Creates directory for wfc
    os.system("mkdir -p " + WFCDIRECTORY)
    print("     Wavefunctions will be saved in directory", WFCDIRECTORY)
    print("     DFT files are in directory", DFTDIRECTORY)
    NPR = d.npr
    print("     This program will run in " + str(NPR) + " processors")
    print()
    NKS = d.nks
    print("     Total number of k-points:", NKS)
    NR1 = d.nr1
    NR2 = d.nr2
    NR3 = d.nr3
    print("     Number of r-points in each direction:", NR1, NR2, NR3)
    NR = d.nr
    print("     Total number of points in real space:", NR)
    NBND = int(d.nbnd)
    print("     Number of bands:", NBND)
    print()
    RPOINT = int(d.rpoint)
    print("     Point choosen for sincronizing phases: ", RPOINT)
    print()

    ##########################################################################
    # Creates files with wfc of bands at nk  ** DFT **
    nk = -1
    nb = -1
    if len(sys.argv) == 1:  # Will run for all k-points and bands
        print("     Will run for all k-points and bands")
        print("     There are", NKS, "k-points and", NBND, "bands.")
        for nk in range(NKS):
            print("     Calculating wfc for k-point", nk)
            dft.wfck2r(nk, 0, NBND - 1)
    elif len(sys.argv) == 2:  # Will run just for k-point nk
        nk = int(sys.argv[1])
        print("     Will run just for k-point", nk)
        print("     There are", NBND, "bands.")
        for nb in range(NBND):
            print("     Calculating wfc for k-point", nk, "and band", nb)
            dft.wfck2r(nk, nb)
    elif len(sys.argv) == 3:  # Will run just for k-point nk and band nb
        nk = int(sys.argv[1])
        nb = int(sys.argv[2])
        print("     Will run just for k-point", nk, "and band", nb)
        print("     Calculating wfc for k-point", nk, "and band", nb)
        dft.wfck2r(nk, nb)
    print()

    ###################################################################################
    # Finished
    footer(contatempo.tempo(STARTTIME, time.time()))
