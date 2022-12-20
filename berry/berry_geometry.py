from multiprocessing import Pool, Array
from typing import Literal

import os
from time import time
import ctypes
import logging

import numpy as np

from berry import log
from berry.utils.jit import numba_njit

try:
    import berry._subroutines.loaddata as d
    import berry._subroutines.loadmeta as m
except:
    pass


def berry_connection(n_pos: int, n_gra: int):
    """
    Calculates the Berry connection.
    """ 
    wfcpos = np.load(os.path.join(m.workdir, f"wfcpos{n_pos}.npy"), mmap_mode="r").conj()
    
    @numba_njit
    def aux_connection() -> np.ndarray:
        """
        Auxiliary function to calculate the Berry connection.
        """
        # Calculation of the Berry connection
        bcc = np.zeros(wfcgra[0].shape, dtype=np.complex128)

        for posi in range(m.nr):
            bcc += 1j * wfcpos[posi] * wfcgra[posi]

        ##  we are assuming that normalization is \sum |\psi|^2 = 1
        ##  if not, needs division by m.nr
        return bcc / m.nr

    start = time()
    bcc = aux_connection()
    logger.info(f"\tberry_connection{n_pos}_{n_gra} calculated in {time() - start:.2f} seconds")


    np.save(os.path.join(m.workdir, f"berryConn{n_pos}_{n_gra}.npy"), bcc)


def berry_curvature(idx: int, idx_: int) -> None:
    """
    Calculates the Berry curvature.
    """
    if idx == idx_:
        wfcgra_ = wfcgra.conj()
    else:
        wfcgra_ = np.load(os.path.join(m.workdir, f"wfcgra{idx_}.npy"), mmap_mode="r").conj()

    @numba_njit
    def aux_curvature() -> np.ndarray:
        """
        Auxiliary function to calculate the Berry curvature.
        """
        bcr = np.zeros(wfcgra[0].shape, dtype=np.complex128)

        for posi in range(m.nr):
            bcr += (
                1j * wfcgra[posi][0] * wfcgra_[posi][1]
                - 1j * wfcgra[posi][1] * wfcgra_[posi][0]
            )
        ##  we are assuming that normalization is \sum |\psi|^2 = 1
        ##  if not, needs division by m.nr
        return bcr / m.nr

    start = time()
    bcr = aux_curvature()
    logger.info(f"\tberry_curvature{idx}_{idx_} calculated in {time() - start:.2f} seconds")

    np.save(os.path.join(m.workdir, f"berryCur{idx}_{idx_}.npy"), bcr)

def run_berry_geometry(max_band: int, min_band: int = 0, npr: int = 1, prop: Literal["curvature", "connection", "both"] = "both", logger_name: str = "geometry", logger_level: int = logging.INFO, flush: bool = False):
    global wfcgra, logger
    logger = log(logger_name, "BERRY GEOMETRY", level=logger_level, flush=flush)
    
    logger.header()
    
    ###########################################################################
    # 1. DEFINING THE CONSTANTS
    ########################################################################### 
    GRA_SIZE  = m.nr * 2 * m.nkx * m.nky
    GRA_SHAPE = (m.nr, 2, m.nkx, m.nky)

    ###########################################################################
    # 2. STDOUT THE PARAMETERS
    ########################################################################### 
    logger.info(f"\tUnique reference of run: {m.refname}")
    logger.info(f"\tProperties to calculate: {prop}")
    logger.info(f"\tMinimum band: {min_band}")
    logger.info(f"\tMaximum band: {max_band}")
    logger.info(f"\tNumber of processes: {npr}\n")

    ###########################################################################
    # 3. CREATE ALL THE ARRAYS
    ###########################################################################
    arr_shell = Array(ctypes.c_double, 2 * GRA_SIZE, lock=False)
    wfcgra    = np.frombuffer(arr_shell, dtype=np.complex128).reshape(GRA_SHAPE)

    ###########################################################################
    # 4. CALCULATE BERRY GEOMETRY
    ###########################################################################
    if prop == "both" or prop == "connection":
        for idx in range(min_band, max_band + 1):
            wfcgra = np.load(os.path.join(m.workdir, f"wfcgra{idx}.npy"))

            work_load = ((idx_pos, idx) for idx_pos in range(min_band, max_band + 1))

            with Pool(npr) as pool:
                pool.starmap(berry_connection, work_load)
    logger.info()

    if prop == "both" or prop == "curvature":
        for idx in range(min_band, max_band + 1):
            wfcgra = np.load(os.path.join(m.workdir, f"wfcgra{idx}.npy"))

            work_load = ((idx, idx_) for idx_ in range(min_band, max_band + 1))

            with Pool(npr) as pool:
                pool.starmap(berry_curvature, work_load)

    ###########################################################################
    # Finished
    ###########################################################################

    logger.footer()

if __name__ == "__main__":
    run_berry_geometry(9, log("berry_geometry", "BERRY GEOMETRY", "version", logging.DEBUG), npr=10, prop="both")