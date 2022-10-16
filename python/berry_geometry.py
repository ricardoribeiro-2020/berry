"""
Calculates the Berry connections and Berry curvatures.
"""

from multiprocessing import Pool, Array
# from functools import partial

import sys
import ctypes

import numpy as np

from cli import berry_props_cli
from jit import numba_njit
from contatempo import time_fn
from log_libs import log

import loaddata as d

args = berry_props_cli()
LOG: log = log("berry_props", "BERRY GEOMETRY", d.version, args["LOG LEVEL"])

###################################################################################
def berry_connection(n_pos: int, n_gra: int):
    """
    Calculates the Berry connection.
    """ 
    wfcpos = np.load(f"wfcpos{n_pos}.npy", mmap_mode="r").conj()
    
    @numba_njit
    def aux_connection() -> np.ndarray:
        """
        Auxiliary function to calculate the Berry connection.
        """
        # Calculation of the Berry connection
        bcc = np.zeros(wfcgra[0].shape, dtype=np.complex128)

        for posi in range(d.nr):
            bcc += 1j * wfcpos[posi] * wfcgra[posi]

        ##  we are assuming that normalization is \sum |\psi|^2 = 1
        ##  if not, needs division by d.nr
        return bcc / d.nr

    bcc = aux_connection()

    np.save(f"berryConn{n_pos}_{n_gra}.npy", bcc)


def berry_curvature(idx: int, idx_: int) -> None:
    """
    Calculates the Berry curvature.
    """
    if idx == idx_:
        wfcgra_ = wfcgra.conj()
    else:
        wfcgra_ = np.load(f"wfcgra{idx_}.npy", mmap_mode="r").conj()

    @numba_njit
    def aux_curvature() -> np.ndarray:
        """
        Auxiliary function to calculate the Berry curvature.
        """
        bcr = np.zeros(wfcgra[0].shape, dtype=np.complex128)

        for posi in range(d.nr):
            bcr += (
                1j * wfcgra[posi][0] * wfcgra_[posi][1]
                - 1j * wfcgra[posi][1] * wfcgra_[posi][0]
            )
        ##  we are assuming that normalization is \sum |\psi|^2 = 1
        ##  if not, needs division by d.nr
        return bcr / d.nr

    bcr = aux_curvature()

    np.save(f"berryCur{idx}_{idx_}.npy", bcr)


if __name__ == "__main__":
    LOG.header()
    
    ###########################################################################
    # 1. DEFINING THE CONSTANTS
    ########################################################################### 
    NPR      = args["NPR"]
    PROP     = args["PROP"]
    MIN_BAND = args["MIN_BAND"]
    MAX_BAND = args["MAX_BAND"]

    GRA_SIZE  = d.nr * 2 * d.nkx * d.nky
    GRA_SHAPE = (d.nr, 2, d.nkx, d.nky)

    ###########################################################################
    # 2. STDOUT THE PARAMETERS
    ########################################################################### 
    LOG.info(f"\tUnique reference of run: {d.refname}")
    LOG.info(f"\tProperties to calculate: {PROP}")
    LOG.info(f"\tMinimum band: {MIN_BAND}")
    LOG.info(f"\tMaximum band: {MAX_BAND}")
    LOG.info(f"\tNumber of processes: {NPR}\n")
    sys.stdout.flush()

    ###########################################################################
    # 3. CREATE ALL THE ARRAYS
    ###########################################################################
    arr_shell = Array(ctypes.c_double, 2 * GRA_SIZE, lock=False)
    wfcgra    = np.frombuffer(arr_shell, dtype=np.complex128).reshape(GRA_SHAPE)

    ###########################################################################
    # 4. CALCULATE BERRY GEOMETRY
    ###########################################################################
    if PROP == "both" or PROP == "connection":
        for idx in range(MIN_BAND, MAX_BAND + 1):
            wfcgra = np.load(f"wfcgra{idx}.npy")

            berry_conn = time_fn(0, 1, prefix="\t")(berry_connection)
            work_load = ((idx_pos, idx) for idx_pos in range(MIN_BAND, MAX_BAND + 1))

            with Pool(NPR) as pool:
                pool.starmap(berry_conn, work_load)

    if PROP == "both" or PROP == "curvature":
        for idx in range(MIN_BAND, MAX_BAND + 1):
            wfcgra = np.load(f"wfcgra{idx}.npy")

            berry_curv = time_fn(0, 1, prefix="\t")(berry_curvature)
            work_load = ((idx, idx_) for idx_ in range(MIN_BAND, MAX_BAND + 1))

            with Pool(NPR) as pool:
                pool.starmap(berry_curv, work_load)

    ###########################################################################
    # Finished
    ###########################################################################

    LOG.footer()
