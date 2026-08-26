from multiprocessing import Pool, Array
from itertools import product
from typing import Dict, Tuple, Sequence

import os, sys
import ctypes
import logging

from findiff import Gradient

import numpy as np

from berry import log
from berry._subroutines.comutator import comute, comute3, comutederiv

try:
    import berry._subroutines.loaddata as d
    import berry._subroutines.loadmeta as m
except (ImportError, FileNotFoundError):
    # data files not created yet (e.g. running the CLI help before preprocessing)
    pass


def load_berry_connections(initial_band: int, conduction_band: int, berry_conn_size: int, berry_conn_shape: Tuple[Sequence[int]]) -> np.ndarray:
    base = Array(ctypes.c_double, berry_conn_size * 2, lock=False)
    berry_connections = np.frombuffer(base, dtype=np.complex128).reshape(berry_conn_shape)

    for i in range(initial_band, conduction_band + 1):
        for j in range(initial_band, conduction_band + 1):
            berry_connections[i - initial_band, j - initial_band] = np.load(os.path.join(m.geometry_dir, f"berryConn{i}_{j}.npy"))

    return berry_connections


def _interpolate_bad_eigenvalues(eigen_array: np.ndarray, bandsfinal: np.ndarray, logger) -> None:
    """Fill eigenvalues at unattributed (-1) k-points (left NaN above) with the average of
    their valid in-BZ neighbours in the same band, and warn. Without this a raw -1 would
    silently index the last band's energy (numpy negative indexing) and corrupt the SHG
    denominators.

    Iterative: each pass fills slots that now have at least one non-NaN same-band neighbour,
    so the fix propagates inward through a clustered bad region. A slot still unreachable when
    propagation stalls (an entire band island with no attributed anchor) is set to 0."""
    band0 = initial_band - m.initial_band        # first bandsfinal column of the band window
    bad = bandsfinal[:, band0:band0 + number_of_bands] < 0
    if not bad.any():
        return
    n_bad = int(bad.sum())
    bad_kp, bad_band = np.where(bad)
    todo = list(zip(bad_kp.tolist(), bad_band.tolist()))
    n_interp = 0
    progress = True
    while todo and progress:
        progress = False
        still = []
        for kp, banda in todo:
            vals = [eigen_array[tuple(int(x) for x in d.nktoijl[nb][:m.dimensions]) + (banda,)]
                    for nb in d.neighbors[kp] if nb >= 0]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                eigen_array[tuple(int(x) for x in d.nktoijl[kp][:m.dimensions]) + (banda,)] = np.mean(vals)
                n_interp += 1
                progress = True
            else:
                still.append((kp, banda))
        todo = still
    for kp, banda in todo:                       # unreachable slots: no anchor anywhere
        eigen_array[tuple(int(x) for x in d.nktoijl[kp][:m.dimensions]) + (banda,)] = 0.0
    msg = (f"\t{n_bad} unattributed (-1) eigenvalue slot(s) over "
           f"{len(set(bad_kp.tolist()))} k-point(s); interpolated {n_interp} from BZ neighbours")
    if todo:
        msg += f", set {len(todo)} to 0 (no resolvable neighbour)"
    logger.warning(msg)


def correct_eigenvalues(bandsfinal: np.ndarray, logger) -> np.ndarray:
    kp = 0
    eigenvalues = d.eigenvalues[:, m.initial_band:] # align to the bandsfinal band numbering
    # bandsfinal columns are 0-based from m.initial_band; the script's band window starts
    # at initial_band (-mb), which may be higher, so offset the column lookup accordingly.
    band0 = initial_band - m.initial_band
    # bandsfinal == -1 marks k-points cluster0 could not attribute; index with NaN here (a raw
    # -1 would silently read the last band) and interpolate from neighbours afterwards.
    def energy(kp, banda):
        bf = bandsfinal[kp, band0 + banda]
        return np.nan if bf < 0 else eigenvalues[kp, bf]
    if m.dimensions == 1:
        eigen_array = np.zeros((m.nkx, number_of_bands))
        for i in range(m.nkx):
            for banda in range(number_of_bands):
                eigen_array[i, banda] = energy(kp, banda)
            kp += 1

    elif m.dimensions == 2:
        eigen_array = np.zeros((m.nkx, m.nky, number_of_bands))
        for j in range(m.nky):
            for i in range(m.nkx):
                for banda in range(number_of_bands):
                    eigen_array[i, j, banda] = energy(kp, banda)
                kp += 1
    else:
        eigen_array = np.zeros((m.nkx, m.nky, m.nkz, number_of_bands))
        for l in range(m.nkz):
            for j in range(m.nky):
                for i in range(m.nkx):
                    for banda in range(number_of_bands):
                        eigen_array[i, j, l, banda] = energy(kp, banda)
                    kp += 1

    _interpolate_bad_eigenvalues(eigen_array, bandsfinal, logger)
    return eigen_array


def get_fermi_delta_ea_grad_ea(grad: Gradient, eigen_array: np.ndarray, conduction_band: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    grad_dea = np.zeros((m.dimensions,) + KSHAPE + (conduction_band + 1, conduction_band + 1), dtype=np.complex128)
    delta_ea = np.zeros(KSHAPE + (conduction_band + 1, conduction_band + 1))
    fermi = np.zeros(KSHAPE + (conduction_band + 1, conduction_band + 1))

    for s, sprime in product(band_list, repeat=2):
        delta_ea[..., s, sprime] = eigen_array[..., s] - eigen_array[..., sprime]
        grad_dea[:, ..., s, sprime] = grad(delta_ea[..., s, sprime])
        if s <= m.vb - initial_band < sprime:
            fermi[..., s, sprime] = 1
        elif sprime <= m.vb - initial_band < s:
            fermi[..., s, sprime] = -1

    return fermi, delta_ea, grad_dea


def precompute_comutederiv_sym(fermi: np.ndarray, logger) -> Dict[Tuple[int, int], np.ndarray]:
    """The generalized-derivative commutator [xi^beta_{s's}, (xi^alpha1_{ss'})_;alpha2]
    does not depend on the photon energy, so it is computed once here instead of at
    every omega.  It is stored already symmetrized in alpha1 <-> alpha2, as required
    by the (alpha1 <-> alpha2) symmetrization of the SHG response (the two driving
    fields are interchangeable), and only for band pairs that can contribute: its
    weight gamma13 is proportional to fermi[s, s'] and vanishes for all other pairs."""
    sym = {}
    for s, sprime in product(band_list, repeat=2):
        if s == sprime or not fermi[..., s, sprime].any():
            continue
        arr = np.empty((m.dimensions,) * 3 + KSHAPE, dtype=np.complex128)
        for beta, alpha1, alpha2 in product(range(m.dimensions), repeat=3):
            arr[beta, alpha1, alpha2] = 0.5 * (
                comutederiv(berry_connections, s, sprime, beta, alpha1, alpha2, m.step)
                + comutederiv(berry_connections, s, sprime, beta, alpha2, alpha1, m.step)
            )
        sym[(s, sprime)] = arr
    if sym:
        size_mb = sum(a.nbytes for a in sym.values()) / 2 ** 20
        logger.info(f"\tPrecomputed symmetrized generalized-derivative commutators for {len(sym)} band pairs ({size_mb:.0f} MB)")
    return sym


def calculate_shg(omega: float, broadening: complex) -> Tuple[float, np.ndarray]:

    omega_ = omega + broadening                                                  # hslash*omega + i*Gamma, in Ry

    gamma1 = CONST * delta_ea / (2 * omega_ - delta_ea)                          # factor dE/g in paper times leading constant
    gamma2 = -fermi / np.square(omega_ - delta_ea)                               # factor f/h^2 in paper (-) to account for change in indices in f and h
    gamma3 = -fermi / (omega_ - delta_ea)                                        # factor f/h in paper (index reference is of h, not f, in equation)
    gamma12 = gamma1 * gamma2
    gamma13 = gamma1 * gamma3

    sig = np.zeros(KSHAPE + (m.dimensions,) * 3, dtype=np.complex128)            # tensor sig_xxx, sig_xxy, ..., sig_yyy

    # beta, alpha1, alpha2 are spatial coordinates
    for beta, alpha1, alpha2 in product(range(m.dimensions), repeat=3):
        for s, sprime in product(band_list, repeat=2):                           # runs through band indices s, s'
            if s == sprime:
                continue

            # term with the gradient of the energy differences ((alpha1 <-> alpha2)-symmetrized)
            sig[..., beta, alpha1, alpha2] += (
                (
                    grad_dea[alpha2][..., s, sprime]
                    * comute(berry_connections, sprime, s, beta, alpha1)
                    + grad_dea[alpha1][..., s, sprime]
                    * comute(berry_connections, sprime, s, beta, alpha2)
                )
                * gamma12[..., s, sprime]
                * 0.5
            )

            # term with the generalized derivative of the Berry connection
            # (precomputed and symmetrized; pairs absent from the dict have gamma13 = 0)
            cd_sym = comutederiv_sym.get((s, sprime))
            if cd_sym is not None:
                sig[..., beta, alpha1, alpha2] += cd_sym[beta, alpha1, alpha2] * gamma13[..., s, sprime]

            # three-band term; after the (alpha1 <-> alpha2) symmetrization the same
            # pair of comute3 products multiplies both occupation factors
            for r in band_list:                                                  # runs through band index r
                if r in (sprime, s):
                    continue
                sig[..., beta, alpha1, alpha2] += (
                    -0.25j
                    * gamma1[..., s, sprime]
                    * (
                        comute3(berry_connections, sprime, s, r, beta, alpha2, alpha1)
                        + comute3(berry_connections, sprime, s, r, beta, alpha1, alpha2)
                    )
                    * (gamma3[..., r, sprime] - gamma3[..., s, r])
                )

    return omega, np.sum(sig, axis=tuple(range(m.dimensions))) * VK * UNIT


def run_shg(conduction_band: int, min_band: int = 0, npr: int = 1, energy_max: float = 2.5, energy_step: float = 0.001, brd: float = 0.01, logger_name: str = "shg", logger_level: int = logging.INFO, flush: bool = False):
    global fermi, delta_ea, grad_dea, comutederiv_sym, band_list, berry_connections, KSHAPE, CONST, VK, UNIT, initial_band, number_of_bands
    logger = log(logger_name, "SECOND HARMONIC GENERATOR", level=logger_level, flush=flush)

    logger.header()

    if min_band > conduction_band:
        logger.error("Error: Minimum band greater than conduction band!")
        logger.footer()
        sys.exit(1)
    if min_band < m.initial_band:
        logger.error(f"Error: Minimum band ({min_band}) below the preprocessing initial band ({m.initial_band})!")
        logger.footer()
        sys.exit(1)

    initial_band = min_band
    number_of_bands = conduction_band - initial_band + 1
    broadening = brd * 1j

    ###########################################################################
    # 1. DEFINING THE CONSTANTS
    ###########################################################################
    RY    = 13.6056923                                                          # Conversion factor from Ry to eV
    VK    = m.step ** m.dimensions                                              # element of volume in k-space (grid units);
                                                                                # the 1/(2pi)^d of the BZ integral is in CONST
    # The k-point grid (and therefore berryConn, grad_dea and VK) is in QE 'tpiba' units,
    # 2pi/alat.  Each SHG term carries three inverse powers of k against the dk^d of the
    # k-sum, leaving d-3 powers of k; convert them to bohr^-1 so that sigma comes out in
    # bohr-based Rydberg atomic units (e^2 = 2, hslash = 1).
    # a1 is in bohr and b1 in 2pi/alat (QE xml conventions), so a1.b1 = alat exactly.
    alat  = float(np.dot(m.a1, m.b1))                                           # lattice parameter (bohr)
    UNIT  = (2 * np.pi / alat) ** (m.dimensions - 3)
    # |e|^3/hslash 1/(2pi)^d in Rydberg units (e^2 = 2 so |e|^3 = 2*sqrt(2));
    # doubled for spin degeneracy unless this is a noncollinear calculation.
    # The -e^3 prefactor of the formula is positive with the signed electron charge e = -|e|.
    if m.noncolin:
        CONST = 2 * np.sqrt(2) / (2 * np.pi) ** m.dimensions
    else:
        CONST = 2 * 2 * np.sqrt(2) / (2 * np.pi) ** m.dimensions

    band_list   = list(range(conduction_band + 1 - initial_band))
    band_info   = list(range(initial_band, conduction_band + 1))

    #TODO: Add docstring with these comments
    # Maximum energy (Ry)
    # Energy step (Ry)
    # energy broadening (Ry)

    cb = conduction_band + 1 - initial_band
    if m.dimensions == 1:
        KSHAPE = (m.nkx,)
    elif m.dimensions == 2:
        KSHAPE = (m.nkx, m.nky)
    else:
        KSHAPE = (m.nkx, m.nky, m.nkz)
    berry_conn_size  = m.dimensions * cb ** 2 * int(np.prod(KSHAPE))
    berry_conn_shape = (cb, cb, m.dimensions) + KSHAPE

    ###########################################################################
    # 2. STDOUT THE PARAMETERS
    ###########################################################################
    logger.info(f"\tUsing {npr} processes")

    logger.info(f"\n\tList of bands: {band_info}")
    logger.info(f"\tNumber of k-points in each direction: {m.nkx} {m.nky} {m.nkz}")
    logger.info(f"\tNumber of bands: {m.nbnd}")
    logger.info(f"\tk-points step, dk {m.step} (2pi/alat units; alat = {alat:.6f} bohr)")     # Defines the step for gradient calculation dk

    logger.info(f"\n\tMaximum energy (Ry): {energy_max}")
    logger.info(f"\tEnergy step (Ry): {energy_step}")
    logger.info(f"\tEnergy broadening (Ry): {np.imag(broadening)}")
    if m.noncolin:
        logger.info(f"\tThis is a noncollinear calculation.")
        logger.info(f"\tConstant e^3/hslash 1/(2pi)^d in Rydberg units: {np.real(CONST)}")
    else:
        logger.info(f"\tThis is a no spin calculation.")
        logger.info(f"\tConstant 2e^3/hslash 1/(2pi)^d in Rydberg units: {np.real(CONST)}")
    logger.info(f"\tNumber of dimensions d = {m.dimensions}")
    logger.info(f"\tVolume (area) element in k space: {VK}; unit conversion to bohr: {UNIT}\n")

    ###########################################################################
    # 3. CREATE ALL THE ARRAYS
    ###########################################################################
    grad = Gradient(h=[m.step] * m.dimensions, acc=2)                            # Defines gradient function in m.dimensions dimensions

    bandsfinal                = np.load(os.path.join(m.data_dir, "bandsfinal.npy"))
    eigen_array               = correct_eigenvalues(bandsfinal, logger)
    berry_connections         = load_berry_connections(initial_band, conduction_band, berry_conn_size, berry_conn_shape)
    fermi, delta_ea, grad_dea = get_fermi_delta_ea_grad_ea(grad, eigen_array, conduction_band - initial_band)
    comutederiv_sym           = precompute_comutederiv_sym(fermi, logger)

    ###########################################################################
    # 4. SECOND HARMONIC GENERATION
    ###########################################################################
    energies = np.arange(0, energy_max + energy_step, energy_step)
    sigma = {}
    with Pool(npr) as pool:
        results = pool.starmap(calculate_shg, ((omega, broadening) for omega in energies))
        for omega, result in results:
            sigma[omega] = result

    ###########################################################################
    # 5. SAVE OUTPUT
    ###########################################################################
    if m.dimensions == 1:
        components = [(0, 0, 0)]
        names = ["xxx"]
    elif m.dimensions == 2:
        components = [(0, 0, 0), (1, 1, 1), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 0, 0)]
        names = ["xxx", "yyy", "xxy", "xyx", "xyy", "yyx", "yxy", "yxx"]
    else:
        components = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 2, 0), (1, 0, 0),
                      (2, 0, 0), (0, 1, 1), (2, 1, 1), (1, 1, 0), (1, 1, 2), (1, 0, 1), (1, 2, 1),
                      (0, 2, 2), (1, 2, 2), (2, 2, 0), (2, 2, 1), (2, 0, 2), (2, 1, 2), (0, 1, 2),
                      (2, 0, 1), (1, 2, 0), (0, 2, 1), (2, 1, 0), (1, 0, 2)]
        names = ["xxx", "yyy", "zzz", "xxy", "xxz", "xyx", "xzx", "yxx",
                 "zxx", "xyy", "zyy", "yyx", "yyz", "yxy", "yzy",
                 "xzz", "yzz", "zzx", "zzy", "zxz", "zyz", "xyz",
                 "zxy", "yzx", "xzy", "zyx", "yxz"]
    header = "# Energy (eV), " + ", ".join(f"sigma_{name}" for name in names)
    header += "  (sigma in Rydberg atomic units: e^2 = 2, hslash = 1, lengths in bohr)\n"

    for filename, part, partname in (("sigma2r.dat", np.real, "Real"), ("sigma2i.dat", np.imag, "Imaginary")):
        with open(os.path.join(m.workdir, filename), "w") as sigm:
            sigm.write(header)
            for omega in energies:
                line = f"{omega * RY:.4f}"
                line += "".join(f"  {part(sigma[omega][c]):.4e}" for c in components)
                sigm.write(line + "\n")
        logger.info(f"\t{partname} part of SHG saved to file {filename}")

    ###################################################################################
    # Finished
    ###################################################################################

    logger.footer()
