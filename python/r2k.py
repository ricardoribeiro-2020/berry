"""
 This program reads a set of wavefunctions for different k and bands and translates that
 to another set for different points in real space,
 the functions become a function of k instead of r
"""
from multiprocessing import Pool, Array
# from functools import partial

import ctypes

from findiff import Gradient

import numpy as np

from cli import r2k_cli
from contatempo import time_fn
from log_libs import log

import loaddata as d

args = r2k_cli()
LOG: log = log("r2k", "R2K", d.version, args["LOG LEVEL"])

# pylint: disable=C0103
###################################################################################
@time_fn(prefix="\t")
def read_wfc_files(banda: int) -> None:
    global read_wfc_kp

    def read_wfc_kp(kp):
        if signalfinal[kp, banda] == -1:                                        # if its a signaled wfc, choose interpolated
            infile = f"{d.wfcdirectory}k0{kp}b0{bandsfinal[kp, banda]}.wfc1"
        else:                                                                   # else choose original
            infile = f"{d.wfcdirectory}k0{kp}b0{bandsfinal[kp, banda]}.wfc"

        wfct_k[:, kp] = np.load(infile)

    with Pool(min(10, NPR)) as pool: #TODO: try to abstract this operation 
        pool.map(read_wfc_kp, range(d.nks))


@time_fn(prefix="\t")
def calculate_wfcpos() -> np.ndarray:
    global calculate_wfcpos_kp

    def calculate_wfcpos_kp(kp):
        wfcpos[kp] = d.phase[kp, d.ijltonk[:, :, 0]] * wfct_k[kp, d.ijltonk[:, :, 0]]

    with Pool(NPR) as pool:
        pool.map(calculate_wfcpos_kp, range(d.nr))


@time_fn(prefix="\t")
def calculate_wfcgra() -> np.ndarray:
    global calculate_wfcgra_kp

    def calculate_wfcgra_kp(kp):
        wfcgra[kp] = grad(wfcpos[kp])

    with Pool(NPR) as pool:
        pool.map(calculate_wfcgra_kp, range(d.nr))


@time_fn(prefix="\t")
def r_to_k(banda: int) -> None:
    read_wfc_files(banda)

    calculate_wfcpos()

    calculate_wfcgra()

    np.save(f"wfcpos{banda}.npy", wfcpos)
    np.save(f"wfcgra{banda}.npy", wfcgra)


if __name__ == "__main__":
    LOG.header()

    ###########################################################################
    # 1. DEFINING THE CONSTANTS
    ########################################################################### 
    NPR = args["NPR"]
    MIN_BAND = args["MIN_BAND"]
    MAX_BAND = args["MAX_BAND"]

    WFCT_K_SHAPE = (d.nr, d.nks)
    WFCPOS_SHAPE = (d.nr, d.nkx, d.nky)
    WFCGRA_SHAPE = (d.nr, 2, d.nkx, d.nky)

    WFCT_K_SIZE = d.nr * d.nks
    WFCPOS_SIZE = d.nr * d.nkx * d.nky
    WFCGRA_SIZE = d.nr * 2 * d.nkx * d.nky

    ###########################################################################
    # 2. STDOUT THE PARAMETERS
    ########################################################################### 
    LOG.info(f"\tUnique reference of run: {d.refname}")
    LOG.info(f"\tNumber of processors to use: {NPR}")
    LOG.info(f"\tMinimum band: {MIN_BAND}")
    LOG.info(f"\tMaximum band: {MAX_BAND}")
    LOG.info(f"\tk-points step, dk: {d.step}")
    LOG.info(f"\tTotal number of k-points: {d.nks}")
    LOG.info(f"\tTotal number of points in real space: {d.nr}")
    LOG.info(f"\tNumber of k-points in each direction: {d.nkx} {d.nky} {d.nkz}")
    LOG.info(f"\tDirectory where the wfc are: {d.wfcdirectory}\n")


    ###########################################################################
    # 3. CREATE ALL THE ARRAYS AND GRADIENT
    ###########################################################################
    grad = Gradient(h=[d.step, d.step], acc=2)                                  # Defines gradient function in 2D
    signalfinal = np.load("signalfinal.npy")
    bandsfinal = np.load("bandsfinal.npy")

    buffer = Array(ctypes.c_double, 2 * WFCT_K_SIZE, lock=False)
    wfct_k = np.frombuffer(buffer, dtype=np.complex128).reshape(WFCT_K_SHAPE)

    buffer = Array(ctypes.c_double, 2 * WFCPOS_SIZE, lock=False)
    wfcpos = np.frombuffer(buffer, dtype=np.complex128).reshape(WFCPOS_SHAPE)

    buffer = Array(ctypes.c_double, 2 * WFCGRA_SIZE, lock=False)
    wfcgra = np.frombuffer(buffer, dtype=np.complex128).reshape(WFCGRA_SHAPE)

    ###########################################################################
    # 4. CALCULATE
    ###########################################################################
    for banda in range(MIN_BAND, MAX_BAND + 1):
        r_to_k(banda)

    ###########################################################################
    # Finished
    ###########################################################################

    LOG.footer()
