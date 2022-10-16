"""  This program reads the wfc from DFT calculations make them coherent and saves
   in separate files

  Can accept 0, 1 or 2 arguments.
  If it has 0 arguments, it will run for all k-points and bands
  If it has 1 argument, it will run just for one k-point, specified by the argument
  If it has 2 arguments, it will run just for 1 k-point and 1 band, specified by the arguments
"""
import os

from cli import generatewfc_cli
from log_libs import log

import dft
import loaddata as d

args = generatewfc_cli()
LOG = log("generatewfc", "GENERATE WFC", d.version, args["LOG LEVEL"])

# pylint: disable=C0103
###################################################################################
if __name__ == "__main__":
    LOG.header()
    os.system("mkdir -p " + d.wfcdirectory)

    ###########################################################################
    # 1. DEFINING THE CONSTANTS
    ###########################################################################
    NK_POINTS = args["NK"]
    BANDS = args["BAND"]

    ###########################################################################
    # 2. STDOUT THE PARAMETERS
    ###########################################################################
    LOG.info(f"\tUnique reference of run: {d.refname}", d.refname)
    LOG.info(f"\tThis program will run in {d.npr} processors")
    LOG.info(f"\tTotal number of k-points: {d.nks}")
    LOG.info(f"\tNumber of bands: {d.nbnd}")
    LOG.info(f"\tNumber of r-points in each direction: {d.nr1} {d.nr2} {d.nr3}")
    LOG.info(f"\tPoint choosen for sincronizing phases: {d.rpoint}")
    LOG.info(f"\tTotal number of points in real space: {d.nr}")
    LOG.info(f"\tDFT files are in {d.dftdirectory}")
    LOG.info(f"\tWavefunctions will be saved in {d.wfcdirectory}\n")

    ###########################################################################
    # 4. WAVEFUNCTIONS EXTRACTION
    ###########################################################################
    if isinstance(NK_POINTS, range):
        LOG.info("\tWill run for all k-points and bands")
        LOG.info(f"\tThere are {d.nks} k-points and {d.nbnd} bands.\n")

        for nk in NK_POINTS:
            LOG.info(f"\tCalculating wfc for k-point {nk}")
            dft._wfck2r(nk, 0, d.nbnd)
    else:
        if isinstance(BANDS, range):
            LOG.info(f"\tWill run just for k-point {NK_POINTS} an all bands.")
            LOG.info(f"\tThere are {d.nks} k-points and {d.nbnd} bands.\n")

            LOG.info(f"\tCalculating wfc for k-point {NK_POINTS}")
            dft._wfck2r(NK_POINTS, 0, d.nbnd)
        else:
            LOG.info(f"\tWill run just for k-point {NK_POINTS} and band {BANDS}.\n")
            dft._wfck2r(NK_POINTS, BANDS)

    ###########################################################################
    # Finished
    ###########################################################################

    LOG.footer()
