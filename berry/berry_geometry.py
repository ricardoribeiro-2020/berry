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
    if m.noncolin:
        wfcpos0 = np.load(os.path.join(m.data_dir, f"wfcpos{n_pos}-0.npy"), mmap_mode="r").conj()
        wfcpos1 = np.load(os.path.join(m.data_dir, f"wfcpos{n_pos}-1.npy"), mmap_mode="r").conj()
    else:
        wfcpos = np.load(os.path.join(m.data_dir, f"wfcpos{n_pos}.npy"), mmap_mode="r").conj()
    
    @numba_njit
    def aux_connection() -> np.ndarray:
        """
        Auxiliary function to calculate the Berry connection.
        """
        # Calculation of the Berry connection
        if m.noncolin:
            bcc = np.zeros(wfcgra0[0].shape, dtype=np.complex128)
            for posi in range(m.nr):
                bcc += 1j * (wfcpos0[posi] * wfcgra0[posi] + wfcpos1[posi] * wfcgra1[posi])
        else:
            bcc = np.zeros(wfcgra[0].shape, dtype=np.complex128)

            for posi in range(m.nr):
                bcc += 1j * wfcpos[posi] * wfcgra[posi]

        ##  we are assuming that normalization is \sum |\psi|^2 = 1
        ##  if not, needs division by m.nr
        return bcc / m.nr

    start = time()
    bcc = aux_connection()
    logger.info(f"\tberry_connection{n_pos}_{n_gra} calculated in {time() - start:.2f} seconds")


    np.save(os.path.join(m.geometry_dir, f"berryConn{n_pos}_{n_gra}.npy"), bcc)


def berry_curvature(idx: int, idx_: int) -> None:
    """
    Calculates the Berry curvature.
    """
    if m.noncolin:
        if idx == idx_:
            wfcgra0_ = wfcgra0.conj()
            wfcgra1_ = wfcgra1.conj()
        else:
            wfcgra0_ = np.load(os.path.join(m.data_dir, f"wfcgra{idx_}-0.npy"), mmap_mode="r").conj()
            wfcgra1_ = np.load(os.path.join(m.data_dir, f"wfcgra{idx_}-1.npy"), mmap_mode="r").conj()
    else:
        if idx == idx_:
            wfcgra_ = wfcgra.conj()
        else:
            wfcgra_ = np.load(os.path.join(m.data_dir, f"wfcgra{idx_}.npy"), mmap_mode="r").conj()

    @numba_njit
    def aux_curvature() -> np.ndarray:
        """
        Auxiliary function to calculate the Berry curvature.
        Attention: this is valid for 2D materials. 
        An expression for 2D Berry curvature is used.
        """
        if m.noncolin:
            bcr = np.zeros(wfcgra0[0].shape, dtype=np.complex128)

            for posi in range(m.nr):
                bcr += (
                    1j * wfcgra0[posi][0] * wfcgra0_[posi][1]
                    - 1j * wfcgra0[posi][1] * wfcgra0_[posi][0]
                    + 1j * wfcgra1[posi][0] * wfcgra1_[posi][1]
                    - 1j * wfcgra1[posi][1] * wfcgra1_[posi][0]
                )
        else:
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

    np.save(os.path.join(m.geometry_dir, f"berryCur{idx}_{idx_}.npy"), bcr)

def run_berry_geometry(max_band: int, min_band: int = 0, npr: int = 1, prop: Literal["curvature", "connection", "both"] = "both", logger_name: str = "geometry", logger_level: int = logging.INFO, flush: bool = False):
    if m.noncolin:
        global wfcgra0, wfcgra1, logger
    else:
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
    if m.noncolin:
        arr_shell0 = Array(ctypes.c_double, 2 * GRA_SIZE, lock=False)
        arr_shell1 = Array(ctypes.c_double, 2 * GRA_SIZE, lock=False)
        wfcgra0    = np.frombuffer(arr_shell0, dtype=np.complex128).reshape(GRA_SHAPE)
        wfcgra1    = np.frombuffer(arr_shell1, dtype=np.complex128).reshape(GRA_SHAPE)
    else:
        arr_shell = Array(ctypes.c_double, 2 * GRA_SIZE, lock=False)
        wfcgra    = np.frombuffer(arr_shell, dtype=np.complex128).reshape(GRA_SHAPE)

    ###########################################################################
    # 4. CALCULATE BERRY GEOMETRY
    ###########################################################################
    if prop == "both" or prop == "connection":
        if m.noncolin:
            for idx in range(min_band, max_band + 1):
                wfcgra0 = np.load(os.path.join(m.data_dir, f"wfcgra{idx}-0.npy"))
                wfcgra1 = np.load(os.path.join(m.data_dir, f"wfcgra{idx}-1.npy"))

                work_load = ((idx_pos, idx) for idx_pos in range(min_band, max_band + 1))

                with Pool(npr) as pool:
                    pool.starmap(berry_connection, work_load)
        else:
            for idx in range(min_band, max_band + 1):
                wfcgra = np.load(os.path.join(m.data_dir, f"wfcgra{idx}.npy"))

                work_load = ((idx_pos, idx) for idx_pos in range(min_band, max_band + 1))

                with Pool(npr) as pool:
                    pool.starmap(berry_connection, work_load)
    logger.info()

    if prop == "both" or prop == "curvature":
        if m.noncolin:
            for idx in range(min_band, max_band + 1):
                wfcgra0 = np.load(os.path.join(m.data_dir, f"wfcgra{idx}-0.npy"))
                wfcgra1 = np.load(os.path.join(m.data_dir, f"wfcgra{idx}-1.npy"))

                work_load = ((idx, idx_) for idx_ in range(min_band, max_band + 1))

                with Pool(npr) as pool:
                    pool.starmap(berry_curvature, work_load)
        else:
            for idx in range(min_band, max_band + 1):
                wfcgra = np.load(os.path.join(m.data_dir, f"wfcgra{idx}.npy"))

                work_load = ((idx, idx_) for idx_ in range(min_band, max_band + 1))

                with Pool(npr) as pool:
                    pool.starmap(berry_curvature, work_load)

    ###########################################################################
    # Finished
    ###########################################################################

    logger.footer()

if __name__ == "__main__":
    run_berry_geometry(9, log("berry_geometry", "BERRY GEOMETRY", "version", logging.DEBUG), npr=10, prop="both")