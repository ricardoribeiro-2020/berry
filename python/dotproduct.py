"""
  This program calculates the dot product of the wfc Bloch factor with their neighbors

"""
from itertools import product
from typing import Tuple, Optional
from multiprocessing import Array, Pool

import os
import sys
import time
import ctypes

import numpy as np

# This are the subroutines and functions
from headerfooter import header, footer
from NestablePool import NestablePool

import contatempo
import loaddata as d

# pylint: disable=C0103
###################################################################################
def _connect(nk, j, neighbor, jNeighbor, dphase, band0, band1):
    "Reads the data from file and "

    with open(f"{d.wfcdirectory}k0{nk}b0{band0}.wfc", "rb") as fichconn:
        wfc0 = np.load(fichconn)
    with open(f"{d.wfcdirectory}k0{neighbor}b0{band1}.wfc", "rb") as fichconn:
        wfc1 = np.load(fichconn)

    dpc[nk, j, band0, band1] = np.sum(dphase * wfc0 * np.conjugate(wfc1)) / d.nr
    dpc[neighbor, jNeighbor, band1, band0] = np.conjugate(dpc[nk, j, band0, band1])


def connection(nkconn, j, neighborconn, jNeighbor, dphaseconn):
    """Calculates the dot product of all combinations of wfc in nkconn and neighborconn."""
    params = {
        "nkconn": (nkconn,),
        "j": (j,),
        "neighborconn": (neighborconn,),
        "jNeighbor": (jNeighbor,),
        "dphaseconn": (dphaseconn,),
        "banda0": range(d.nbnd),
        "banda1": range(d.nbnd),
    }

    with Pool(1) as pool: #TODO Add Some sort of logic behind the number of cores
        pool.starmap(_connect, product(*params.values()))


def pre_connection(
    nk: int, j: int, neighbor: int, jNeighbor: Tuple[np.ndarray]
) -> None:

    dphase = d.phase[:, nk] * np.conjugate(d.phase[:, neighbor])

    print("      Calculating   nk = " + str(nk) + "  neighbor = " + str(neighbor))
    sys.stdout.flush()

    connection(nk, j, neighbor, jNeighbor, dphase)


def generate_pre_connection_args(
    nk: int, j: int
) -> Optional[Tuple[int, int, int, Tuple[np.ndarray]]]:
    """Generates the arguments for the pre_connection function."""
    neighbor = d.neighbors[nk, j]
    if neighbor != -1 and neighbor > nk:
        jNeighbor = np.where(d.neighbors[neighbor] == nk)
        return (nk, j, neighbor, jNeighbor)
    return None


###################################################################################
if __name__ == "__main__":
    header("DOTPRODUCT", d.version, time.asctime())

    STARTTIME = time.time()  # Starts counting time
    DPC_SIZE = d.nks * 4 * d.nbnd * d.nbnd
    DPC_SHAPE = (d.nks, 4, d.nbnd, d.nbnd)
    RUN_PARAMS = {
        "nk_points": range(d.nks),
        "num_neibhors": range(4),  # TODO Fix Hardcoded value
    }
    NPR = 20
    if "-np" in sys.argv:
        NPR = int(sys.argv[sys.argv.index("-np") + 1])
    if NPR > os.cpu_count():
        print("Warning: Number of processors is greater than the number of available processors")
        print("Setting number of processors to the number of available processors")
        if input("Do you want to continue? ([y]/n)? ") == "n":
            sys.exit(0)
        NPR = int(os.cpu_count() / 2) # Hyperthreading is not considered
    SUFFIX = "" # Append to the output file name - for dubbuging purposes
    if "-o" in sys.argv:
        SUFFIX = sys.argv[sys.argv.index("-o") + 1]

    # Reading data needed for the run

    print("     Unique reference of run:", d.refname)
    print("     Directory where the wfc are:", d.wfcdirectory)
    print("     Total number of k-points:", d.nks)
    print("     Total number of points in real space:", d.nr)
    print("     Number of processors to use", NPR)
    print("     Number of bands:", d.nbnd)
    print()
    print("     Phases loaded")
    print("     Neighbors loaded")

    # Finished reading data needed for the run
    print()
    ##########################################################

    # Creating a buffer for the dpc np.ndarray
    dpc_base = Array(ctypes.c_double, 2 * DPC_SIZE, lock=False)
    # Initializing shared instance of np.ndarray 'dpc'
    dpc = np.frombuffer(dpc_base, dtype=complex).reshape(DPC_SHAPE)

    ##########################################################

    # Creating a list of tuples with the neighbors of each k-point
    #! A Bigger Pool here seems to be better 
    #TODO Add some sort of logic behind the number of cores
    with NestablePool(NPR) as pool:
        pre_connection_args = list(filter(None, pool.starmap(generate_pre_connection_args, product(*RUN_PARAMS.values()))))
        pool.starmap(pre_connection, pre_connection_args)

    dp = np.abs(dpc)

    # Save dot products to file
    with open(f"dpc{SUFFIX}.npy", "wb") as fich:
        np.save(fich, dpc)
    print(f"     Dot products saved to file dpc{SUFFIX}.npy")

    # Save dot products modulus to file
    with open(f"dp{SUFFIX}.npy", "wb") as fich:
        np.save(fich, dp)
    print(f"     Dot products modulus saved to file dp{SUFFIX}.npy")

    ###################################################################################
    # Finished
    footer(contatempo.tempo(STARTTIME, time.time()))
