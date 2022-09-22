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

from contatempo import tempo
from headerfooter import header, footer

import dft
import loaddata as d

# pylint: disable=C0103
###################################################################################
if __name__ == "__main__":
    header("GENERATEWFC", d.version, time.asctime())

    STARTTIME = time.time()  # Starts counting time

    WFCDIRECTORY = str(d.wfcdirectory)
    DFTDIRECTORY = str(d.dftdirectory)

    # Creates directory for wfc
    os.system("mkdir -p " + WFCDIRECTORY)
    print("     Unique reference of run:", d.refname)
    print("     Wavefunctions will be saved in directory", WFCDIRECTORY)
    print("     DFT files are in directory", DFTDIRECTORY)
    print("     This program will run in " + str(d.npr) + " processors")
    print()
    print("     Total number of k-points:", d.nks)
    print("     Number of r-points in each direction:", d.nr1, d.nr2, d.nr3)
    print("     Total number of points in real space:", d.nr)
    print("     Number of bands:", d.nbnd)
    print()
    print("     Point choosen for sincronizing phases: ", d.rpoint)
    print()

    ##########################################################################
    # Creates files with wfc of bands at nk  ** DFT **
    nk = -1
    nb = -1
    if len(sys.argv) == 1:  # Will run for all k-points and bands
        print("     Will run for all k-points and bands")
        print("     There are", d.nks, "k-points and", d.nbnd, "bands.")
        for nk in range(d.nks):
            print("     Calculating wfc for k-point", nk)
            dft._wfck2r(nk, 0, d.nbnd - 1)
    elif len(sys.argv) == 2:  # Will run just for k-point nk
        nk = int(sys.argv[1])
        print("     Will run just for k-point", nk)
        print("     There are", d.nbnd, "bands.")
        for nb in range(d.nbnd):
            print("     Calculating wfc for k-point", nk, "and band", nb)
            dft._wfck2r(nk, nb)
    elif len(sys.argv) == 3:  # Will run just for k-point nk and band nb
        nk = int(sys.argv[1])
        nb = int(sys.argv[2])
        print("     Will run just for k-point", nk, "and band", nb)
        print("     Calculating wfc for k-point", nk, "and band", nb)
        dft._wfck2r(nk, nb)
    print()

    ###################################################################################
    # Finished
    footer(tempo(STARTTIME, time.time()))
