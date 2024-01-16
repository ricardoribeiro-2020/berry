from multiprocessing import Pool, Array

import os
from time import time
import ctypes
import logging

from findiff import Gradient

import numpy as np

from berry import log

try:
    import berry._subroutines.loaddata as d
    import berry._subroutines.loadmeta as m
except:
    pass


def read_wfc_files(banda: int, npr: int) -> None:
    global read_wfc_kp

    def read_wfc_kp(kp):
        if m.noncolin:
            infile0  = f"{m.wfcdirectory}/k0{kp}b0{bandsfinal[kp, banda]}-0.wfc"
            infile1  = f"{m.wfcdirectory}/k0{kp}b0{bandsfinal[kp, banda]}-1.wfc"
            wfct_k0[:, kp] = np.load(infile0)
            wfct_k1[:, kp] = np.load(infile1)
        else:
            if signalfinal[kp, banda] == -1:                                        # if its a signaled wfc, choose corrected
                infile = f"{m.wfcdirectory}/k0{kp}b0{bandsfinal[kp, banda]}.wfc1"
            else:                                                                   # else choose original
                infile = f"{m.wfcdirectory}/k0{kp}b0{bandsfinal[kp, banda]}.wfc"

            wfct_k[:, kp] = np.load(infile)

    with Pool(min(10, npr)) as pool: #TODO: try to abstract this operation 
        pool.map(read_wfc_kp, range(m.nks))


def calculate_wfcpos(npr: int) -> np.ndarray:
    global calculate_wfcpos_kp

    def calculate_wfcpos_kp(kp):
        if m.noncolin:
            wfcpos0[kp] = d_phase[kp, d.ijltonk[:, :, :]] * wfct_k0[kp, d.ijltonk[:, :, :]]
            wfcpos1[kp] = d_phase[kp, d.ijltonk[:, :, :]] * wfct_k1[kp, d.ijltonk[:, :, :]]
        else:
            wfcpos[kp] = d_phase[kp, d.ijltonk[:, :, :]] * wfct_k[kp, d.ijltonk[:, :, :]]

    with Pool(npr) as pool:
        pool.map(calculate_wfcpos_kp, range(m.nr))


def calculate_wfcgra(npr: int) -> np.ndarray:
    global calculate_wfcgra_kp

    def calculate_wfcgra_kp(kp):
        if m.noncolin:
            wfcgra0[kp] = grad(wfcpos0[kp])
            wfcgra1[kp] = grad(wfcpos1[kp])

        else:
            wfcgra[kp] = grad(wfcpos[kp])

    with Pool(npr) as pool:
        pool.map(calculate_wfcgra_kp, range(m.nr))


def r_to_k(banda: int, npr: int) -> None:
    start = time()
    read_wfc_files(banda, npr)
    logger.debug(f"\twfc files read in {time() - start:.2f} seconds")

    start = time()
    calculate_wfcpos(npr)
    logger.debug(f"\twfcpos{banda} calculated in {time() - start:.2f} seconds")

    start = time()
    calculate_wfcgra(npr)
    logger.debug(f"\twfcgra{banda} calculated in {time() - start:.2f} seconds")

    start = time()
    #IDEA: Try saving this files into a folder in different chunks
    if m.noncolin:
        np.save(os.path.join(m.data_dir, f"wfcpos{banda}-0.npy"), wfcpos0)
        np.save(os.path.join(m.data_dir, f"wfcpos{banda}-1.npy"), wfcpos1)
        np.save(os.path.join(m.data_dir, f"wfcgra{banda}-0.npy"), wfcgra0)
        np.save(os.path.join(m.data_dir, f"wfcgra{banda}-1.npy"), wfcgra1)
    else:
        np.save(os.path.join(m.data_dir, f"wfcpos{banda}.npy"), wfcpos)
        np.save(os.path.join(m.data_dir, f"wfcgra{banda}.npy"), wfcgra)

    logger.debug(f"\twfcpos{banda} and wfcgra{banda} saved in {time() - start:.2f} seconds\n")


def run_r2k(max_band: int, npr: int = 1, min_band: int = 0, logger_name: str = "r2k", logger_level: int = logging.INFO, flush: bool = False):
    if m.noncolin:
        global grad, signalfinal, bandsfinal, wfct_k0, wfct_k1, wfcpos0, wfcpos1, wfcgra0, wfcgra1, logger, d_phase
    else:
        global grad, signalfinal, bandsfinal, wfct_k, wfcpos, wfcgra, logger, d_phase

    logger = log(logger_name, "R2K", level=logger_level, flush=flush)

    logger.header()

    ###########################################################################
    # 1. DEFINING THE CONSTANTS
    ########################################################################### 

    WFCT_K_SHAPE = (m.nr, m.nks)
    if m.dimensions == 1:
        WFCPOS_SHAPE = (m.nr, m.nkx)
        WFCGRA_SHAPE = (m.nr, m.dimensions, m.nkx)
    elif m.dimensions == 2:
        WFCPOS_SHAPE = (m.nr, m.nkx, m.nky)
        WFCGRA_SHAPE = (m.nr, m.dimensions, m.nkx, m.nky)
    else:
        WFCPOS_SHAPE = (m.nr, m.nkx, m.nky, m.nkz)
        WFCGRA_SHAPE = (m.nr, m.dimensions, m.nkx, m.nky, m.nkz)

    WFCT_K_SIZE = m.nr * m.nks
    WFCPOS_SIZE = m.nr * m.nkx * m.nky * m.nkz
    WFCGRA_SIZE = m.nr * m.dimensions * m.nkx * m.nky * m.nkz

    ###########################################################################
    # 2. STDOUT THE PARAMETERS
    ########################################################################### 
    logger.info(f"\tUnique reference of run: {m.refname}")
    logger.info(f"\tNumber of processors to use: {npr}")
    logger.info(f"\tMinimum band: {min_band}")
    logger.info(f"\tMaximum band: {max_band}")
    logger.info(f"\tk-points step, dk: {m.step}")
    logger.info(f"\tTotal number of k-points: {m.nks}")
    logger.info(f"\tTotal number of points in real space: {m.nr}")
    logger.info(f"\tNumber of k-points in each direction: {m.nkx} {m.nky} {m.nkz}")
    logger.info(f"\tDirectory where the wfc are: {m.wfcdirectory}\n")
    logger.info(f"\n\t{m.dimensions} dimensions calculation.")

    ###########################################################################
    # 3. CREATE ALL THE ARRAYS AND GRADIENT
    ###########################################################################
    if m.dimensions == 1:                   # Defines gradient function in 1, 2 and 3D
        grad = Gradient(h=[m.step], acc=2)
    elif m.dimensions == 2:
        grad = Gradient(h=[m.step, m.step], acc=2) 
    else:
        grad = Gradient(h=[m.step, m.step, m.step], acc=2)

    signalfinal = np.load(os.path.join(m.data_dir, "signalfinal.npy"))
    bandsfinal = np.load(os.path.join(m.data_dir, "bandsfinal.npy"))
    d_phase = np.load(os.path.join(m.data_dir, "phase.npy"))
    logger.info(f"\tSignal and bands files loaded")

    if m.noncolin:
        buffer = Array(ctypes.c_double, 2 * WFCT_K_SIZE, lock=False)
        wfct_k0 = np.frombuffer(buffer, dtype=np.complex128).reshape(WFCT_K_SHAPE)
        wfct_k1 = np.frombuffer(buffer, dtype=np.complex128).reshape(WFCT_K_SHAPE)

        buffer = Array(ctypes.c_double, 2 * WFCPOS_SIZE, lock=False)
        wfcpos0 = np.frombuffer(buffer, dtype=np.complex128).reshape(WFCPOS_SHAPE)
        wfcpos1 = np.frombuffer(buffer, dtype=np.complex128).reshape(WFCPOS_SHAPE)

        buffer = Array(ctypes.c_double, 2 * WFCGRA_SIZE, lock=False)
        wfcgra0 = np.frombuffer(buffer, dtype=np.complex128).reshape(WFCGRA_SHAPE)
        wfcgra1 = np.frombuffer(buffer, dtype=np.complex128).reshape(WFCGRA_SHAPE)

    else:
        buffer = Array(ctypes.c_double, 2 * WFCT_K_SIZE, lock=False)
        wfct_k = np.frombuffer(buffer, dtype=np.complex128).reshape(WFCT_K_SHAPE)

        buffer = Array(ctypes.c_double, 2 * WFCPOS_SIZE, lock=False)
        wfcpos = np.frombuffer(buffer, dtype=np.complex128).reshape(WFCPOS_SHAPE)

        buffer = Array(ctypes.c_double, 2 * WFCGRA_SIZE, lock=False)
        wfcgra = np.frombuffer(buffer, dtype=np.complex128).reshape(WFCGRA_SHAPE)


    ###########################################################################
    # 4. CALCULATE
    ###########################################################################
    for banda in range(min_band, max_band + 1):
        r_to_k(banda, npr)

    ###########################################################################
    # Finished
    ###########################################################################
    logger.footer()

if __name__ == "__main__":
    run_r2k(9, log("r2k", "R2K", "version", logging.DEBUG), 20)


    