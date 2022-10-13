"""
  This program calculates the dot product of the wfc Bloch factor with their neighbors

"""
from typing import Tuple
from multiprocessing import Array, Pool

import sys
import time
import ctypes

import numpy as np

from cli import dotproduct_cli
from jit import numba_njit
from headerfooter import header, footer
from contatempo import time_fn

import contatempo
import loaddata as d

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

    print(header("DOTPRODUCT", d.version, time.asctime()))
    STARTTIME = time.time()  # Starts counting time

    ###########################################################################
    # 1. DEFINING THE CONSTANTS
    ########################################################################### 
    NPR = args["NPR"]

    DPC_SIZE = d.nks * 4 * d.nbnd * d.nbnd
    DPC_SHAPE = (d.nks, 4, d.nbnd, d.nbnd)

    ###########################################################################
    # 2. STDOUT THE PARAMETERS
    ########################################################################### 
    print(f"\tUnique reference of run: {d.refname}")
    print(f"\tNumber of processors to use: {NPR}")
    print(f"\tNumber of bands: {d.nbnd}")
    print(f"\tTotal number of k-points: {d.nks}")
    print(f"\tTotal number of points in real space: {d.nr}")
    print(f"\tDirectory where the wfc are: {d.wfcdirectory}\n")
    sys.stdout.flush()

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
    dpc /=d.nr
    dp = np.abs(dpc)

    ###########################################################################
    # 5. SAVE OUTPUT
    ###########################################################################
    np.save("dpc.npy", dpc)
    np.save("dp.npy", dp)
    print(f"\n\tDot products saved to file dpc.npy")
    print(f"\tDot products modulus saved to file dp.npy")

    ###########################################################################
    # Finished
    ###########################################################################

    print(footer(contatempo.tempo(STARTTIME, time.time())))
