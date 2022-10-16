"""
  This program calculates the dot product of the wfc Bloch factor with their neighbors

"""
from typing import Tuple
from multiprocessing import Array, Pool

import ctypes

import numpy as np

from cli import dotproduct_cli
from log_libs import log
from contatempo import time_fn

import loaddata as d

LOG: log = log("dotproduct", "DOT PRODUCT", d.version)

# pylint: disable=C0103
###################################################################################
@time_fn(0, 2, prefix="\t")
def dot(nk: int, j: int, neighbor: int, jNeighbor: Tuple[np.ndarray]) -> None:

    dphase = d.phase[:, nk] * d.phase[:, neighbor].conj()

    for band0 in range(d.nbnd):
        wfc0 = np.load(f"{d.wfcdirectory}k0{nk}b0{band0}.wfc")
        for band1 in range(d.nbnd):
            wfc1 = np.load(f"{d.wfcdirectory}k0{neighbor}b0{band1}.wfc").conj()

            dpc[nk, j, band0, band1] = np.einsum("k,k,k->", dphase, wfc0, wfc1)
            dpc[neighbor, jNeighbor, band1, band0] = dpc[nk, j, band0, band1].conj()


def get_point_neighbors(nk: int, j: int) -> None:
    """Generates the arguments for the pre_connection function."""
    neighbor = d.neighbors[nk, j]
    if neighbor != -1 and neighbor > nk:
        jNeighbor = np.where(d.neighbors[neighbor] == nk)

        return (nk, j, neighbor, jNeighbor)
    return None


###################################################################################
if __name__ == "__main__":
    args = dotproduct_cli()

    LOG.header()

    ###########################################################################
    # 1. DEFINING THE CONSTANTS
    ########################################################################### 
    NPR = args["NPR"]

    DPC_SIZE = d.nks * 4 * d.nbnd * d.nbnd
    DPC_SHAPE = (d.nks, 4, d.nbnd, d.nbnd)

    ###########################################################################
    # 2. STDOUT THE PARAMETERS
    ########################################################################### 
    LOG.info(f"\tUnique reference of run: {d.refname}")
    LOG.info(f"\tNumber of processors to use: {NPR}")
    LOG.info(f"\tNumber of bands: {d.nbnd}")
    LOG.info(f"\tTotal number of k-points: {d.nks}")
    LOG.info(f"\tTotal number of points in real space: {d.nr}")
    LOG.info(f"\tDirectory where the wfc are: {d.wfcdirectory}\n")

    ###########################################################################
    # 3. CREATE ALL THE ARRAYS
    ###########################################################################
    dpc_base = Array(ctypes.c_double, 2 * DPC_SIZE, lock=False)
    dpc = np.frombuffer(dpc_base, dtype=np.complex128).reshape(DPC_SHAPE)
    dp = np.zeros(DPC_SHAPE, dtype=np.float64)

    ###########################################################################
    # 4. CALCULATE THE CONDUCTIVITY
    ###########################################################################
    with Pool(NPR) as pool:
        pre_connection_args = (
            args
            for nk in range(d.nks)
            for j in range(4)
            if (args := get_point_neighbors(nk, j)) is not None
        )
        pool.starmap(dot, pre_connection_args)
    dpc /= d.nr
    dp = np.abs(dpc)

    ###########################################################################
    # 5. SAVE OUTPUT
    ###########################################################################
    np.save("dpc.npy", dpc)
    np.save("dp.npy", dp)
    LOG.info(f"\tDot products saved to file dpc.npy")
    LOG.info(f"\tDot products modulus saved to file dp.npy")

    ###########################################################################
    # Finished
    ###########################################################################

    LOG.footer()
