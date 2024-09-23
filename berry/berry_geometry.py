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
        try:
            wfcpos0 = np.load(os.path.join(m.data_dir, f"wfcpos{n_pos}-0.npy"), mmap_mode="r").conj()
            wfcpos1 = np.load(os.path.join(m.data_dir, f"wfcpos{n_pos}-1.npy"), mmap_mode="r").conj()
        except:
            wfcpos0 = np.load(os.path.join(m.data_dir, f"wfcpos{n_pos}-0.npy")).conj()
            wfcpos1 = np.load(os.path.join(m.data_dir, f"wfcpos{n_pos}-1.npy")).conj()

        @numba_njit
        def aux_connection() -> np.ndarray:
            """
            Auxiliary function to calculate the Berry connection.
            """
            # Calculation of the Berry connection
            bcc = np.zeros(wfcgra0[0].shape, dtype=np.complex128)
            for posi in range(m.nr):
                bcc += 1j * (wfcpos0[posi] * wfcgra0[posi] + wfcpos1[posi] * wfcgra1[posi])

            ##  we are assuming that normalization is \sum |\psi|^2 = 1
            ##  if not, needs division by m.nr
            return bcc / m.nr
    else:
        try:
            wfcpos = np.load(os.path.join(m.data_dir, f"wfcpos{n_pos}.npy"), mmap_mode="r").conj()
        except:
            wfcpos = np.load(os.path.join(m.data_dir, f"wfcpos{n_pos}.npy")).conj()
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


    np.save(os.path.join(m.geometry_dir, f"berryConn{n_pos}_{n_gra}.npy"), bcc)


def chern_number(curv) -> None:
    chern = 0
    if m.dimensions == 2:
        chern = np.sum(curv) * (np.linalg.norm(m.b1) / m.nkx) * (np.linalg.norm(m.b2) / m.nky) / (2 * np.pi)
    else:  # 3D 
        chern = (np.sum(curv[0]) * np.linalg.norm(m.b1) / m.nkx
               + np.sum(curv[1]) * np.linalg.norm(m.b2) / m.nky
               + np.sum(curv[2]) * np.linalg.norm(m.b3) / m.nkz) / (2 * np.pi)

    return chern

def berry_curvature(idx: int, idx_: int) -> None:
    """
    Calculates the Berry curvature.
    """
    if m.noncolin:
        if idx == idx_:
            wfcgra0_ = wfcgra0.conj()
            wfcgra1_ = wfcgra1.conj()
        else:
            try:
                wfcgra0_ = np.load(os.path.join(m.data_dir, f"wfcgra{idx_}-0.npy"), mmap_mode="r").conj()
                wfcgra1_ = np.load(os.path.join(m.data_dir, f"wfcgra{idx_}-1.npy"), mmap_mode="r").conj()
            except:
                wfcgra0_ = np.load(os.path.join(m.data_dir, f"wfcgra{idx_}-0.npy")).conj()
                wfcgra1_ = np.load(os.path.join(m.data_dir, f"wfcgra{idx_}-1.npy")).conj()

        if m.dimensions == 2:                # 2D case
            @numba_njit
            def aux_curvature() -> np.ndarray:
                """
                Auxiliary function to calculate the Berry curvature.
                Attention: this is valid for 2D and 3D materials.
                """
                bcr = np.zeros(wfcgra0[0].shape, dtype=np.complex128)
                for posi in range(m.nr):
                    bcr += (
                        1j * wfcgra0[posi][1] * wfcgra0_[posi][0]
                        - 1j * wfcgra0[posi][0] * wfcgra0_[posi][1]
                        + 1j * wfcgra1[posi][1] * wfcgra1_[posi][0]
                        - 1j * wfcgra1[posi][0] * wfcgra1_[posi][1]
                    )

                ##  we are assuming that normalization is \sum |\psi|^2 = 1
                ##  if not, needs division by m.nr
                return bcr / m.nr
            
        else:                                # 3D case
            @numba_njit
            def aux_curvature():
                """
                Auxiliary function to calculate the Berry curvature.
                Attention: this is valid for 2D and 3D materials.
                """ 
                bcr0 = np.zeros(wfcgra0[0].shape, dtype=np.complex128)
                bcr1 = np.zeros(wfcgra0[0].shape, dtype=np.complex128)
                bcr2 = np.zeros(wfcgra0[0].shape, dtype=np.complex128)
                for posi in range(m.nr):
                    bcr0 += (
                        1j * wfcgra0[posi][2] * wfcgra0_[posi][1]
                        - 1j * wfcgra0[posi][1] * wfcgra0_[posi][2]
                        + 1j * wfcgra1[posi][2] * wfcgra1_[posi][1]
                        - 1j * wfcgra1[posi][1] * wfcgra1_[posi][2]
                    )
                    bcr1 += (
                        1j * wfcgra0[posi][0] * wfcgra0_[posi][2]
                        - 1j * wfcgra0[posi][2] * wfcgra0_[posi][0]
                        + 1j * wfcgra1[posi][0] * wfcgra1_[posi][2]
                        - 1j * wfcgra1[posi][2] * wfcgra1_[posi][0]
                    )
                    bcr2 += (
                        1j * wfcgra0[posi][1] * wfcgra0_[posi][0]
                        - 1j * wfcgra0[posi][0] * wfcgra0_[posi][1]
                        + 1j * wfcgra1[posi][1] * wfcgra1_[posi][0]
                        - 1j * wfcgra1[posi][0] * wfcgra1_[posi][1]
                    )
    
                ##  we are assuming that normalization is \sum |\psi|^2 = 1
                ##  if not, needs division by m.nr
                return bcr0 / m.nr,  bcr1 / m.nr, bcr2 / m.nr
    else:
        if idx == idx_:
            wfcgra_ = wfcgra.conj()
        else:
            try:
                wfcgra_ = np.load(os.path.join(m.data_dir, f"wfcgra{idx_}.npy"), mmap_mode="r").conj()
            except:
                wfcgra_ = np.load(os.path.join(m.data_dir, f"wfcgra{idx_}.npy")).conj()
        if m.dimensions == 2:                # 2D case
            @numba_njit
            def aux_curvature() -> np.ndarray:
                """
                Auxiliary function to calculate the Berry curvature.
                Attention: this is valid for 2D and 3D materials.
                """

                bcr = np.zeros(wfcgra[0].shape, dtype=np.complex128)
                for posi in range(m.nr):
                    bcr += (
                        1j * wfcgra[posi][0] * wfcgra_[posi][1]
                        - 1j * wfcgra[posi][1] * wfcgra_[posi][0]
                    )
            
                ##  we are assuming that normalization is \sum |\psi|^2 = 1
                ##  if not, needs division by m.nr
                return bcr
            
        else:                                # 3D case
            @numba_njit
            def aux_curvature():
                """
                Auxiliary function to calculate the Berry curvature.
                Attention: this is valid for 2D and 3D materials.
                """
                bcr0 = np.zeros(wfcgra[0].shape, dtype=np.complex128)
                bcr1 = np.zeros(wfcgra[0].shape, dtype=np.complex128)
                bcr2 = np.zeros(wfcgra[0].shape, dtype=np.complex128)
                for posi in range(m.nr):
                    bcr0 += (
                        1j * wfcgra[posi][2] * wfcgra_[posi][1]
                        - 1j * wfcgra[posi][1] * wfcgra_[posi][2]
                    )
                    bcr1 += (
                        1j * wfcgra[posi][0] * wfcgra_[posi][2]
                        - 1j * wfcgra[posi][2] * wfcgra_[posi][0]
                    )
                    bcr2 += (
                        1j * wfcgra[posi][1] * wfcgra_[posi][0]
                        - 1j * wfcgra[posi][0] * wfcgra_[posi][1]
                    )

                ##  we are assuming that normalization is \sum |\psi|^2 = 1
                ##  if not, needs division by m.nr
                return bcr0 / m.nr,  bcr1 / m.nr, bcr2 / m.nr

    start = time()

    bcr = aux_curvature() if m.dimensions == 2 else np.array(aux_curvature())
    logger.info(f"\tberry_curvature{idx}_{idx_} calculated in {time() - start:.2f} seconds")

    np.save(os.path.join(m.geometry_dir, f"berryCur{idx}_{idx_}.npy"), bcr)

    if idx == idx_:
        chern_num[idx] = chern_number(bcr)
    

def run_berry_geometry(max_band: int, min_band: int = 0, npr: int = 1, prop: Literal["curv", "conn", "both", "chern"] = "both", logger_name: str = "geometry", logger_level: int = logging.INFO, flush: bool = False):
    if m.noncolin:
        global wfcgra0, wfcgra1, chern_num, logger
    else:
        global wfcgra, chern_num, logger

    logger = log(logger_name, "BERRY GEOMETRY", level=logger_level, flush=flush)

    logger.header()

    ###########################################################################
    # 1. DEFINING THE CONSTANTS
    ###########################################################################
    if m.dimensions == 1:
        GRA_SIZE  = m.nr * m.dimensions * m.nkx
        GRA_SHAPE = (m.nr, m.dimensions , m.nkx)
    elif m.dimensions == 2:
        GRA_SIZE  = m.nr * m.dimensions * m.nkx * m.nky
        GRA_SHAPE = (m.nr, m.dimensions, m.nkx, m.nky)
    else:
        GRA_SIZE  = m.nr * m.dimensions * m.nkx * m.nky * m.nkz
        GRA_SHAPE = (m.nr, m.dimensions, m.nkx, m.nky, m.nkz)

    chern_num = np.zeros((max_band + 1), dtype=np.complex128)

    ###########################################################################
    # 2. STDOUT THE PARAMETERS
    ###########################################################################
    logger.info(f"\tUnique reference of run: {m.refname}")
    logger.info(f"\tProperties to calculate: {prop}")
    logger.info(f"\tMinimum band: {min_band}")
    logger.info(f"\tMaximum band: {max_band}")
    logger.info(f"\tNumber of processes: {npr}\n")
    logger.info(f"\t{m.dimensions} dimensions calculation.\n")

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
    if prop == "both" or prop == "conn":
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
    if m.dimensions == 1:
        logger.info(f"\tBerry curvature is not defined for 1D materials.")
    else:
        if prop == "both" or prop == "curv":
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
            np.save(os.path.join(m.geometry_dir, "chern_number.npy"), chern_num)
            logger.info(f"\tchern_number.npy saved")

        if prop == "chern":
            for idx in range(min_band, max_band + 1):
                curv = np.load(os.path.join(m.geometry_dir, f"berryCur{idx}_{idx}.npy"))
                chern_num[idx] = chern_number(curv)   
            np.save(os.path.join(m.geometry_dir, "chern_number.npy"), chern_num)
            logger.info(f"\tchern_number.npy saved")


    ###########################################################################
    # Finished
    ###########################################################################

    logger.footer()

if __name__ == "__main__":
    run_berry_geometry(9, log("berry_geometry", "BERRY GEOMETRY", "version", logging.DEBUG), npr=10, prop="both")
