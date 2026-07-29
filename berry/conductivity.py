from multiprocessing import Array, Pool
from typing import Tuple, Sequence
from itertools import product

import os, sys
import ctypes
import logging

import numpy as np # type: ignore

from berry import log

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
    silently index the last band's energy (numpy negative indexing) and corrupt the
    conductivity denominators.

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
    # bandsfinal == -1 marks k-points cluster could not attribute; index with NaN here (a raw
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

def get_delta_eigen_array_and_fermi(eigen_array: np.ndarray, conduction_band: int) -> Tuple[np.ndarray, np.ndarray]:
    delta_eigen_array = np.zeros(KSHAPE + (conduction_band + 1, conduction_band + 1))
    fermi = np.zeros(KSHAPE + (conduction_band + 1, conduction_band + 1))

    for s in band_list:
        for sprime in band_list:
            delta_eigen_array[..., s, sprime] = eigen_array[..., s] - eigen_array[..., sprime]
            if s <= m.vb - initial_band < sprime:
                fermi[..., s, sprime] = 1
            elif sprime <= m.vb - initial_band < s:
                fermi[..., s, sprime] = -1
    return delta_eigen_array, fermi

def compute_conductivity(omega: float, delta_eigen_array: np.ndarray, fermi: np.ndarray, broadening: complex) -> Tuple[float, np.ndarray]:
    gamma = CONST * delta_eigen_array / (omega + broadening - delta_eigen_array) # factor that multiplies
    sig = np.zeros((m.dimensions, m.dimensions), dtype=np.complex128)            # matrix sig_xx, sig_xy, sig_yy, sig_yx, etc

    for s in band_list:                                                          # runs through index s
        for sprime in band_list:                                                 # runs through index s'
            if s == sprime:
                continue
            for beta in range(m.dimensions):                                     # beta is spatial coordinate
                for alpha in range(m.dimensions):                                # alpha is spatial coordinate

                    sig[alpha, beta] += np.sum(
                        gamma[..., sprime, s]
                        * berry_connections[s][sprime][alpha]
                        * berry_connections[sprime][s][beta]
                        * fermi[..., s, sprime]
                    )

    return (omega, sig * VK * UNIT)

#TODO: ADD assertions to all functions in order to check if the inputs are correct
#IDEA: Maybe create a type checking decorator (USE pydantic)
def run_conductivity(conduction_band: int, npr: int = 1, min_band: int = 0, energy_max: float = 2.5, energy_step: float = 0.001, brd: float = 0.01, logger_name: str = "conductivity", logger_level: int = logging.INFO, flush: bool = False):
    global band_list, berry_connections, KSHAPE, CONST, VK, UNIT, initial_band, number_of_bands
    logger = log(logger_name, "CONDUCTIVITY", level=logger_level, flush=flush)
    # conduction_band is the number of the highest conduction band to consider.

    logger.header()

    initial_band = min_band

    if initial_band > conduction_band:
        logger.error("Error: Minimum band greater than conduction band!")
        logger.footer()
        sys.exit(1)
    if initial_band < m.initial_band:
        logger.error(f"Error: Minimum band ({initial_band}) below the preprocessing initial band ({m.initial_band})!")
        logger.footer()
        sys.exit(1)
    number_of_bands = conduction_band - initial_band + 1
    broadening = brd * 1j
    ###########################################################################
    # 1. DEFINING THE CONSTANTS
    ###########################################################################
    RY    = 13.6056923                                                          # Conversion factor from Ry to eV
    VK    = m.step ** m.dimensions                                              # element of volume in k-space (grid units);
                                                                                # the 1/(2pi)^d of the BZ integral is in CONST
    # The k-point grid (and therefore berryConn and VK) is in QE 'tpiba' units, 2pi/alat.
    # The k-sum of xi^alpha xi^beta dk^d leaves d-2 powers of k; convert them to bohr^-1
    # so that sigma comes out in bohr-based Rydberg atomic units (e^2 = 2, hslash = 1).
    # a1 is in bohr and b1 in 2pi/alat (QE xml conventions), so a1.b1 = alat exactly.
    alat  = float(np.dot(m.a1, m.b1))                                           # lattice parameter (bohr)
    UNIT  = (2 * np.pi / alat) ** (m.dimensions - 2)
    # i e^2/hslash 1/(2pi)^d in Rydberg units (e^2 = 2);
    # doubled for spin degeneracy unless this is a noncolinear calculation
    if m.noncolin:
        CONST = 2j / (2 * np.pi) ** m.dimensions
    else:
        CONST = 2 * 2j / (2 * np.pi) ** m.dimensions

    band_list   = list(range(conduction_band + 1 - initial_band))
    band_info   = list(range(initial_band, conduction_band + 1))

    #TODO: add function docstring with these comments
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
        logger.info(f"\tThis is a noncolinear calculation.")
        logger.info(f"\tConstant e^2/hslash 1/(2pi)^d in Rydberg units: {np.imag(CONST)}")
    else:
        logger.info(f"\tThis is a no spin calculation.")
        logger.info(f"\tConstant 2e^2/hslash 1/(2pi)^d in Rydberg units: {np.imag(CONST)}")
    logger.info(f"\tNumber of dimensions d = {m.dimensions}")
    logger.info(f"\tVolume (area) element in k space: {VK}; unit conversion to bohr: {UNIT}\n")

    ###########################################################################
    # 3. CREATE ALL THE ARRAYS
    ###########################################################################
    bandsfinal               = np.load(os.path.join(m.data_dir, "bandsfinal.npy"))
    eigen_array              = correct_eigenvalues(bandsfinal, logger)
    berry_connections        = load_berry_connections(initial_band, conduction_band, berry_conn_size, berry_conn_shape)
    delta_eigen_array, fermi = get_delta_eigen_array_and_fermi(eigen_array, conduction_band - initial_band)

    ###########################################################################
    # 4. CALCULATE THE CONDUCTIVITY
    ###########################################################################
    energies = np.arange(0, energy_max + energy_step, energy_step)
    with Pool(npr) as pool:
        work_load = product(energies, [delta_eigen_array], [fermi], [broadening])
        sigma = dict(pool.starmap(compute_conductivity, work_load))

    ###########################################################################
    # 5. SAVE OUTPUT
    ###########################################################################
    if m.dimensions == 1:
        components = [(0, 0)]
        header = "# Energy (eV), sigma"
    elif m.dimensions == 2:
        components = [(0, 0), (1, 1), (1, 0), (0, 1)]
        header = "# Energy (eV), sigma_xx,  sigma_yy,  sigma_yx,  sigma_xy"
    else:
        components = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        header = ("# Energy (eV), sigma_xx,  sigma_yy,  sigma_zz,  sigma_xy, sigma_xz,"
                  "  sigma_yx,  sigma_yz,  sigma_zx,  sigma_zy")
    header += "  (sigma in Rydberg atomic units: e^2 = 2, hslash = 1, lengths in bohr)\n"

    for filename, part, partname in (("sigmar.dat", np.real, "Real"), ("sigmai.dat", np.imag, "Imaginary")):
        with open(os.path.join(m.workdir, filename), "w") as sigm:
            sigm.write(header)
            for omega in energies:
                line = f"{omega * RY:.4f}"
                line += "".join(f"  {part(sigma[omega][alpha, beta]):.4e}" for alpha, beta in components)
                sigm.write(line + "\n")
        logger.info(f"\t{partname} part of conductivity saved to file {filename}")

    ###########################################################################
    # Finished
    ###########################################################################
    logger.footer()
