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
from cli import generatewfc_cli
from headerfooter import header, footer

import dft
import loaddata as d

# pylint: disable=C0103
###################################################################################
if __name__ == "__main__":
    args = generatewfc_cli()

    header("GENERATEWFC", d.version, time.asctime())
    os.system("mkdir -p " + d.wfcdirectory)
    STARTTIME = time.time()

    ###########################################################################
    # 1. DEFINING THE CONSTANTS
    ###########################################################################
    NK_POINTS = args["NK"]
    BANDS = args["BAND"]

    ###########################################################################
    # 2. STDOUT THE PARAMETERS
    ###########################################################################
    print(f"\tUnique reference of run: {d.refname}", d.refname)
    print(f"\tThis program will run in {d.npr} processors")
    print(f"\tTotal number of k-points: {d.nks}")
    print(f"\tNumber of bands: {d.nbnd}")
    print(f"\tNumber of r-points in each direction: {d.nr1} {d.nr2} {d.nr3}")
    print(f"\tPoint choosen for sincronizing phases: {d.rpoint}")
    print(f"\tTotal number of points in real space: {d.nr}")
    print(f"\tDFT files are in {d.dftdirectory}")
    print(f"\tWavefunctions will be saved in {d.wfcdirectory}\n")
    sys.stdout.flush()

    ###########################################################################
    # 4. SECONG HARMONIC GENERATION
    ###########################################################################
    if isinstance(NK_POINTS, range):
        print("\tWill run for all k-points and bands")
        print(f"\tThere are {d.nks} k-points and {d.nbnd} bands.\n")

        for nk in NK_POINTS:
            print(f"\tCalculating wfc for k-point {nk}")
            dft._wfck2r(nk, 0, d.nbnd - 1)
    else:
        if isinstance(BANDS, range):
            print(f"\tWill run just for k-point {NK_POINTS}.")
            print(f"\tThere are {d.nks} k-points and {d.nbnd} bands.\n")
        else:
            print(f"\tWill run just for k-point {NK_POINTS} and band {BANDS[0]}.\n")

        for band in BANDS:
            print(f"\tCalculating wfc for k-point {NK_POINTS} and band {band}")
            dft._wfck2r(NK_POINTS, band)

    ###########################################################################
    # Finished
    ###########################################################################

    footer(tempo(STARTTIME, time.time()))
