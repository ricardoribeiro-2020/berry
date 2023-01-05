from multiprocessing import Pool, Array
from typing import Tuple

import os
from time import time
import ctypes
import logging

import numpy as np

from berry import log

try:
    import berry._subroutines.loaddata as d
    import berry._subroutines.loadmeta as m
except:
    pass


#TODO: Figure out how to share the dpc array between processes inside a class
def dot(nk: int, j: int, neighbor: int, jNeighbor: Tuple[np.ndarray]) -> None:
    start = time()

    dphase = d_phase[:, nk] * d_phase[:, neighbor].conj()

    for band0 in range(m.nbnd):
        wfc0 = np.load(os.path.join(m.wfcdirectory, f"k0{nk}b0{band0}.wfc"))
        for band1 in range(m.nbnd):
            wfc1 = np.load(os.path.join(m.wfcdirectory, f"k0{neighbor}b0{band1}.wfc")).conj()

            dpc[nk, j, band0, band1] = np.einsum("k,k,k->", dphase, wfc0, wfc1)
            dpc[neighbor, jNeighbor, band1, band0] = dpc[nk, j, band0, band1].conj()

    logger.debug(f"\tFinished of nk: {nk:>4}\tneighbor: {neighbor:>4}\tin: {(time() - start):>4.2f} seconds")


def get_point_neighbors(nk: int, j: int) -> None:
    """Generates the arguments for the pre_connection function."""
    neighbor = d.neighbors[nk, j]
    if neighbor != -1 and neighbor > nk:
        jNeighbor = np.where(d.neighbors[neighbor] == nk)

        return (nk, j, neighbor, jNeighbor)
    return None

def run_dot(npr: int = 1, logger_name: str = "dot", logger_level: logging = logging.INFO, flush: bool = False):
    global dpc, logger, d_phase
    logger = log(logger_name, "DOT PRODUCT", level=logger_level, flush=flush)

    if not 0 < npr <= os.cpu_count():
        raise ValueError(f"npr must be between 1 and {os.cpu_count()}")

    logger.header()

    ###########################################################################
    # 1. DEFINING THE CONSTANTS
    ########################################################################### 
    DPC_SIZE = m.nks * 4 * m.nbnd * m.nbnd
    DPC_SHAPE = (m.nks, 4, m.nbnd, m.nbnd)

    ###########################################################################
    # 2. STDOUT THE PARAMETERS
    ########################################################################### 
    logger.info(f"\tUnique reference of run: {m.refname}")
    logger.info(f"\tNumber of processors to use: {npr}")
    logger.info(f"\tNumber of bands: {m.nbnd}")
    logger.info(f"\tTotal number of k-points: {m.nks}")
    logger.info(f"\tTotal number of points in real space: {m.nr}")
    logger.info(f"\tDirectory where the wfc are: {m.wfcdirectory}\n")

    ###########################################################################
    # 3. CREATE ALL THE ARRAYS
    ###########################################################################
    dpc_base = Array(ctypes.c_double, 2 * DPC_SIZE, lock=False)
    dpc = np.frombuffer(dpc_base, dtype=np.complex128).reshape(DPC_SHAPE)
    dp = np.zeros(DPC_SHAPE, dtype=np.float64)
    d_phase = np.load(os.path.join(m.workdir, "phase.npy"))

    ###########################################################################
    # 4. CALCULATE THE CONDUCTIVITY
    ###########################################################################
    with Pool(npr) as pool:
        pre_connection_args = (
            args
            for nk in range(m.nks)
            for j in range(4)
            if (args := get_point_neighbors(nk, j)) is not None
        )
        pool.starmap(dot, pre_connection_args)
    dpc /= m.nr
    dp = np.abs(dpc)

    ###########################################################################
    # 5. SAVE OUTPUT
    ###########################################################################
    np.save(os.path.join(m.workdir, "dpc.npy"), dpc)
    np.save(os.path.join(m.workdir, "dp.npy"), dp)
    logger.info(f"\n\tDot products saved to file dpc.npy")
    logger.info(f"\tDot products modulus saved to file dp.npy")

    ###########################################################################
    # Finished
    ###########################################################################
    logger.footer()

if __name__ == "__main__":
    run_dot(log("dotproduct", "DOT PRODUCT", "version"), 20)