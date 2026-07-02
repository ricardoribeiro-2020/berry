from multiprocessing import Pool, Array
from typing import Literal

import os
from time import time
import ctypes
import logging

import numpy as np

from berry import log
from berry.utils.jit import numba_njit
#from berry._subroutines.comutator import deriv
from berry.shg import load_berry_connections

try:
    import berry._subroutines.loaddata as d
    import berry._subroutines.loadmeta as m
except:
    pass


def deriv(berryConnection, s, sprime, alpha1, alpha2, dk):
    """ Derivative of the Berry connection."""
    from findiff import Gradient
    if m.dimensions == 1:
        grad = Gradient(h=[dk], acc=4)  # Defines gradient function in 1D
    elif m.dimensions == 2:
        grad = Gradient(h=[dk, dk], acc=4)  # Defines gradient function in 2D
    else:
        grad = Gradient(h=[dk, dk, dk], acc=4)  # Defines gradient function in 3D
        
    a = grad(berryConnection[s][sprime][alpha1])


    return a[alpha2]

def loadz(pathz, path, mmap_mode=None):
    if os.path.exists(pathz): # reading .npz compressed file if exists
        r = np.load(pathz, mmap_mode=mmap_mode)
        r = r['arr_0']
    else: # otherwise read .npy file
        r = np.load(path, mmap_mode=mmap_mode)
    
    return r

def berry_connection(n_pos: int, n_gra: int):
    """
    Calculates the Berry connection.
    """
    if m.noncolin:
        try:
            wfcpos0 = loadz(os.path.join(m.data_dir, f"wfcpos{n_pos}-0.npz"), os.path.join(m.data_dir, f"wfcpos{n_pos}-0.npy"), mmap_mode="r").conj()
            wfcpos1 = loadz(os.path.join(m.data_dir, f"wfcpos{n_pos}-1.npz"), os.path.join(m.data_dir, f"wfcpos{n_pos}-1.npy"), mmap_mode="r").conj()
        except:
            wfcpos0 = loadz(os.path.join(m.data_dir, f"wfcpos{n_pos}-0.npz"), os.path.join(m.data_dir, f"wfcpos{n_pos}-0.npy")).conj()
            wfcpos1 = loadz(os.path.join(m.data_dir, f"wfcpos{n_pos}-1.npz"), os.path.join(m.data_dir, f"wfcpos{n_pos}-1.npy")).conj()

        @numba_njit
        def aux_connection() -> np.ndarray:
            """
            Auxiliary function to calculate the Berry connection.
            """
            # Calculation of the Berry connection
            bcc = np.zeros(wfcgra0[0].shape, dtype=np.complex128)
            for posi in range(m.nr):
                bcc += 1j * (wfcpos0[posi] * wfcgra0[posi] + wfcpos1[posi] * wfcgra1[posi])

            ##  normalization convention: (1/nr) * sum_r |u|^2 = 1
            ##  (the .wfc files hold the periodic parts u_nk; see docs/berry_geometry_physics.md)
            return bcc / m.nr
    else:
        try:
            wfcpos = loadz(os.path.join(m.data_dir, f"wfcpos{n_pos}.npz"), os.path.join(m.data_dir, f"wfcpos{n_pos}.npy"), mmap_mode="r").conj()
        except:
            wfcpos = loadz(os.path.join(m.data_dir, f"wfcpos{n_pos}.npz"), os.path.join(m.data_dir, f"wfcpos{n_pos}.npy")).conj()
        @numba_njit
        def aux_connection() -> np.ndarray:
            """
            Auxiliary function to calculate the Berry connection.
            """
            # Calculation of the Berry connection
            bcc = np.zeros(wfcgra[0].shape, dtype=np.complex128)
            for posi in range(m.nr):
                bcc += 1j * wfcpos[posi] * wfcgra[posi]

            ##  normalization convention: (1/nr) * sum_r |u|^2 = 1
            ##  (the .wfc files hold the periodic parts u_nk; see docs/berry_geometry_physics.md)
            return bcc / m.nr

    start = time()
    bcc = aux_connection()
    logger.info(f"\tberry_connection{n_pos}_{n_gra} calculated in {time() - start:.2f} seconds")


    np.save(os.path.join(m.geometry_dir, f"berryConn{n_pos}_{n_gra}.npy"), bcc)


def chern_number(curv):
    """Flux of the Berry curvature through the k-mesh, divided by 2*pi.

    The mesh is square with spacing m.step in every direction (its actual
    construction), so the Riemann area element is step^2 -- NOT |b|/nk, which
    is only equal to step when the mesh happens to tile the BZ exactly.  The
    result is the Chern number when the mesh spans the full BZ; for a partial
    patch it is the (gauge-invariant) patch flux / 2*pi.

    3D: returns the three-component Chern vector.  C_alpha is a 2D integral
    over a slice perpendicular to axis alpha, averaged over the n_alpha
    slices.  (The old form summed each component over the whole 3D mesh times
    a single 1D spacing -- dimensionally inconsistent.)
    """
    if m.dimensions == 1:
        logger.warning("\tChern number is not defined for 1D materials; returning 0.")
        return 0
    if m.dimensions == 2:
        return np.sum(curv) * m.step**2 / (2 * np.pi)
    # 3D: per-slice average of the flux of each curvature component
    return np.array([np.sum(curv[0]) * m.step**2 / m.nkx,
                     np.sum(curv[1]) * m.step**2 / m.nky,
                     np.sum(curv[2]) * m.step**2 / m.nkz]) / (2 * np.pi)

def berry_phase(pos, x, y):
    """Gauge-invariant Berry phase of the counterclockwise plaquette at (x, y),
    with the standard sign: bp = +Omega * dk^2 in the continuum limit.

    The discrete link <u_k|u_{k+dk}> has angle -A.dk (since <u|du> = -iA with
    A = i<u|grad u>), so the counterclockwise product of THESE links gives
    -flux; the minus sign restores the standard orientation.  Verified against
    the gauge-invariant perturbation formula on a C=+1 model."""
    bp = (np.sum(pos[:, x, y].conj()     * pos[:, x+1, y])
       *  np.sum(pos[:, x+1, y].conj()   * pos[:, x+1, y+1])
       *  np.sum(pos[:, x+1, y+1].conj() * pos[:, x, y+1])
       *  np.sum(pos[:, x, y+1].conj()   * pos[:, x, y])) / (m.nr ** 4)

    bp = -np.angle(bp)

    return bp

def chern_number_bp(pos) -> np.complex128:
    """Fukui-Hatsugai-Suzuki lattice Chern number: sum of plaquette Berry
    phases / 2*pi.  Each plaquette phase is gauge invariant.  Note: the loops
    cover the open (nkx-1)x(nky-1) rectangle of plaquettes -- the wrap-around
    row/column is not included, so the total is exactly integer only when the
    mesh tiles the closed BZ torus (for a partial patch it is the patch flux)."""
    chern = 0
    if m.dimensions == 2:
        for i in range(m.nkx -1):
            for j in range(m.nky -1):
                chern += berry_phase(pos, i, j)
        chern /= (2*np.pi)
    else:
        logger.warning("\tchern_bp is only implemented for 2D materials; returning 0.")

    return chern

def chern_number_bp_bz(pos) -> np.complex128:
    """Berry phase of the counterclockwise Wilson loop around the mesh
    boundary, / 2*pi, with the standard sign (= +flux of Omega through the
    mesh, mod 1 -- Arg is only defined mod 2*pi, so this cannot distinguish C
    from C+-1; use chern_bp for the integer).  Each link is normalized to unit
    modulus before entering the product: the result stays fully gauge
    invariant (intermediate gauge phases cancel in the product) and the float
    underflow of a long product of |overlap|<1 factors is gone (the old code
    needed np.complex256 for that, which no longer exists in NumPy 2)."""
    chern = 0.0
    if m.dimensions == 2:
        def _link(z):
            return z / abs(z)
        prod = 1.0 + 0.0j
        for i in range(m.nkx-1):
            prod *= _link(np.sum(pos[:, i, 0].conj() * pos[:, i+1, 0]))
            prod *= _link(np.sum(pos[:, i, m.nky-1] * pos[:, i+1, m.nky-1].conj()))
        for j in range(m.nky-1):
            prod *= _link(np.sum(pos[:, m.nkx-1, j].conj() * pos[:, m.nkx-1, j+1]))
            prod *= _link(np.sum(pos[:, 0, j] * pos[:, 0, j+1].conj()))
        # links <u_k|u_next> carry angle -A.dk (see berry_phase): negate for
        # the standard orientation
        chern = -np.angle(prod) / (2*np.pi)
    else:
        logger.warning("\tchern_bp_bz is only implemented for 2D materials; returning 0.")

    return chern



def berry_curvature(idx: int, idx_: int) -> None:
    """
    Calculates the Berry curvature.
    """
    if m.dimensions == 1:
        logger.warning("\tBerry curvature is not defined for 1D materials; skipping.")
        return
    if m.noncolin:
        if idx == idx_:
            wfcgra0_ = wfcgra0.conj()
            wfcgra1_ = wfcgra1.conj()
        else:
            try:
                wfcgra0_ = loadz(os.path.join(m.data_dir, f"wfcgra{idx_}-0.npz"), os.path.join(m.data_dir, f"wfcgra{idx_}-0.npy"), mmap_mode="r").conj()
                wfcgra1_ = loadz(os.path.join(m.data_dir, f"wfcgra{idx_}-1.npz"), os.path.join(m.data_dir, f"wfcgra{idx_}-1.npy"), mmap_mode="r").conj()
            except:
                wfcgra0_ = loadz(os.path.join(m.data_dir, f"wfcgra{idx_}-0.npz"), os.path.join(m.data_dir, f"wfcgra{idx_}-0.npy")).conj()
                wfcgra1_ = loadz(os.path.join(m.data_dir, f"wfcgra{idx_}-1.npz"), os.path.join(m.data_dir, f"wfcgra{idx_}-1.npy")).conj()

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

                ##  normalization convention: (1/nr) * sum_r |u|^2 = 1
                ##  (the .wfc files hold the periodic parts u_nk; see docs/berry_geometry_physics.md)
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
    
                ##  normalization convention: (1/nr) * sum_r |u|^2 = 1
                ##  (the .wfc files hold the periodic parts u_nk; see docs/berry_geometry_physics.md)
                return bcr0 / m.nr,  bcr1 / m.nr, bcr2 / m.nr
    else:
        if idx == idx_:
            wfcgra_ = wfcgra.conj()
        else:
            try:
                wfcgra_ = loadz(os.path.join(m.data_dir, f"wfcgra{idx_}.npz"), os.path.join(m.data_dir, f"wfcgra{idx_}.npy"), mmap_mode="r").conj()
            except:
                wfcgra_ = loadz(os.path.join(m.data_dir, f"wfcgra{idx_}.npz"), os.path.join(m.data_dir, f"wfcgra{idx_}.npy")).conj()

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
            
                ##  normalization convention: (1/nr) * sum_r |u|^2 = 1
                ##  (the .wfc files hold the periodic parts u_nk; see docs/berry_geometry_physics.md)
                return bcr / m.nr
            
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

                ##  normalization convention: (1/nr) * sum_r |u|^2 = 1
                ##  (the .wfc files hold the periodic parts u_nk; see docs/berry_geometry_physics.md)
                return bcr0 / m.nr,  bcr1 / m.nr, bcr2 / m.nr

    start = time()

    bcr = aux_curvature() if m.dimensions == 2 else np.array(aux_curvature())
    # Standard sign convention Omega = curl(A) with A = i<u|grad u> (bra on the
    # FIRST band index).  The kernels above conjugate the SECOND band (idx_),
    # which yields -Omega* of the standard matrix; converting here on the small
    # (nk-mesh) result keeps the kernels' memory pattern (no extra nr-sized
    # conjugated copies).  Diagonal elements come out real and equal to
    # -2 Im<d1 u|d2 u>, matching the plaquette (chern_bp) orientation.
    bcr = -np.conj(bcr)
    logger.info(f"\tberry_curvature{idx}_{idx_} calculated in {time() - start:.2f} seconds")

    np.save(os.path.join(m.geometry_dir, f"berryCur{idx}_{idx_}.npy"), bcr)


def berry_curvature_curl(idx: int, idx_: int, berry_connection) -> None:
    """
    Calculates the Berry curvature using the curl of Berry connections.
    """
    if m.dimensions == 1:
        logger.warning("\tBerry curvature (curl) is not defined for 1D materials; skipping.")
        return

    # deriv(bc, s, s', alpha1, alpha2, dk) = d_{alpha2} A_{alpha1}, so the curl
    # Omega_c = d_a A_b - d_b A_a is deriv(..., b, a, ...) - deriv(..., a, b, ...).
    # (The old code had TWO bugs here: the second term was a dangling expression
    # statement -- never subtracted -- and the term order gave -Omega.)
    if m.dimensions == 2:                # 2D case
        def aux_curvature() -> np.ndarray:
            """
            Auxiliary function to calculate the Berry curvature.
            Attention: this is valid for 2D and 3D materials.
            """

            bcr = (deriv(berry_connection, idx, idx_, 1, 0, m.step)
                   - deriv(berry_connection, idx, idx_, 0, 1, m.step))

            return bcr

    else:                                # 3D case
        def aux_curvature():
            """
            Auxiliary function to calculate the Berry curvature.
            Attention: this is valid for 2D and 3D materials.
            """

            bcr0 = (deriv(berry_connection, idx, idx_, 2, 1, m.step)
                    - deriv(berry_connection, idx, idx_, 1, 2, m.step))

            bcr1 = (deriv(berry_connection, idx, idx_, 0, 2, m.step)
                    - deriv(berry_connection, idx, idx_, 2, 0, m.step))

            bcr2 = (deriv(berry_connection, idx, idx_, 1, 0, m.step)
                    - deriv(berry_connection, idx, idx_, 0, 1, m.step))

            return bcr0, bcr1, bcr2

    start = time()

    bcr = aux_curvature() if m.dimensions == 2 else np.array(aux_curvature())
    logger.info(f"\tberry_curvature{idx}_{idx_}_curl calculated in {time() - start:.2f} seconds")

    np.save(os.path.join(m.geometry_dir, f"berryCur{idx}_{idx_}_curl.npy"), bcr)
    
    return bcr
    

def run_berry_geometry(max_band: int, min_band: int = 0, npr: int = 1, prop: Literal["curv", "conn", "both", "chern", "chern_curl", "chern_bp", "chern_bp_bz"] = "both", digits: int = 0, logger_name: str = "geometry", logger_level: int = logging.INFO, flush: bool = False):
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

    # 2D: one Chern number per band; 3D: a three-component Chern vector per band
    if m.dimensions == 3:
        chern_num = np.zeros((max_band + 1, 3), dtype=np.complex128)
    else:
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
                wfcgra0 = loadz(os.path.join(m.data_dir, f"wfcgra{idx}-0.npz"), os.path.join(m.data_dir, f"wfcgra{idx}-0.npy"))
                wfcgra1 = loadz(os.path.join(m.data_dir, f"wfcgra{idx}-1.npz"), os.path.join(m.data_dir, f"wfcgra{idx}-1.npy"))

                work_load = ((idx_pos, idx) for idx_pos in range(min_band, max_band + 1))

                with Pool(npr) as pool:
                    pool.starmap(berry_connection, work_load)
        else:
            for idx in range(min_band, max_band + 1):
                wfcgra = loadz(os.path.join(m.data_dir, f"wfcgra{idx}.npz"), os.path.join(m.data_dir, f"wfcgra{idx}.npy"))

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
                    wfcgra0 = loadz(os.path.join(m.data_dir, f"wfcgra{idx}-0.npz"), os.path.join(m.data_dir, f"wfcgra{idx}-0.npy"))
                    wfcgra1 = loadz(os.path.join(m.data_dir, f"wfcgra{idx}-1.npz"), os.path.join(m.data_dir, f"wfcgra{idx}-1.npy"))

                    work_load = ((idx, idx_) for idx_ in range(min_band, max_band + 1))

                    with Pool(npr) as pool:
                        pool.starmap(berry_curvature, work_load)
            else:
                for idx in range(min_band, max_band + 1):
                    wfcgra = loadz(os.path.join(m.data_dir, f"wfcgra{idx}.npz"), os.path.join(m.data_dir, f"wfcgra{idx}.npy"))

                    work_load = ((idx, idx_) for idx_ in range(min_band, max_band + 1))

                    with Pool(npr) as pool:
                        pool.starmap(berry_curvature, work_load)

        if prop == "chern":
            for idx in range(min_band, max_band + 1):
                curv = np.load(os.path.join(m.geometry_dir, f"berryCur{idx}_{idx}.npy"))
                chern_num[idx] = chern_number(curv)

            # save the chern number
            chern_num = np.round(np.real(chern_num), decimals=digits)
            np.save(os.path.join(m.geometry_dir, "chern_number.npy"), chern_num)
            logger.info(f"\tchern_number.npy saved")
            logger.info(f"\n{'Band:' : >13} Chern Number")
            for ind, val in enumerate(chern_num):
                logger.info(f"{f'{ind}:' : >13} {val}")

        if prop == "chern_curl":
            number_of_bands = max_band + 1 - min_band
            if m.dimensions == 2:
                berry_conn_size  = m.dimensions * m.nkx * m.nky * (number_of_bands) ** 2
                berry_conn_shape = (number_of_bands, number_of_bands, m.dimensions, m.nkx, m.nky)
            else:
                berry_conn_size  = m.dimensions * m.nkx * m.nky * m.nkz * (number_of_bands) ** 2
                berry_conn_shape = (number_of_bands, number_of_bands, m.dimensions, m.nkx, m.nky, m.nkz)
            berry_connections = load_berry_connections(max_band, berry_conn_size, berry_conn_shape, min_band)
            for idx in range(min_band, max_band + 1):
                curv = berry_curvature_curl(idx, idx, berry_connections)
                chern_num[idx] = chern_number(curv)   
            np.save(os.path.join(m.geometry_dir, "chern_number_curl.npy"), chern_num)
            logger.info(f"\tchern_number_curl.npy saved")

        if prop == "chern_bp":
            for idx in range(min_band, max_band + 1):
                pos = loadz(os.path.join(m.data_dir, f"wfcpos{idx}.npz"), os.path.join(m.data_dir, f"wfcpos{idx}.npy"))
                chern_num[idx] = chern_number_bp(pos)
            np.save(os.path.join(m.geometry_dir, "chern_number_bp.npy"), chern_num)
            logger.info(f"\tchern_number_bp.npy saved")

        if prop == "chern_bp_bz":
            for idx in range(min_band, max_band + 1):
                pos = loadz(os.path.join(m.data_dir, f"wfcpos{idx}.npz"), os.path.join(m.data_dir, f"wfcpos{idx}.npy"))
                chern_num[idx] = chern_number_bp_bz(pos)   
            np.save(os.path.join(m.geometry_dir, "chern_number_bp_bz.npy"), chern_num)
            logger.info(f"\tchern_number_bp_bz.npy saved")




    ###########################################################################
    # Finished
    ###########################################################################

    logger.footer()

if __name__ == "__main__":
    run_berry_geometry(9, log("berry_geometry", "BERRY GEOMETRY", "version", logging.DEBUG), npr=10, prop="both")
