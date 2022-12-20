from multiprocessing import Array, Pool
from typing import Tuple, Sequence
from itertools import product

import os
import ctypes
import logging

import numpy as np

from berry import log

try:
    import berry._subroutines.loaddata as d
    import berry._subroutines.loadmeta as m
except:
    pass


def load_berry_connections(conduction_band: int, berry_conn_size: int, berry_conn_shape: Tuple[Sequence[int]]) -> np.ndarray:
    base = Array(ctypes.c_double, berry_conn_size * 2, lock=False)
    berry_connections = np.frombuffer(base, dtype=np.complex128).reshape(berry_conn_shape)

    for i in range(conduction_band + 1):
        for j in range(conduction_band + 1):
            berry_connections[i, j] = np.load(os.path.join(m.workdir, f"berryConn{i}_{j}.npy"))

    return berry_connections


def correct_eigenvalues(bandsfinal: np.ndarray) -> np.ndarray:
    kp = 0
    eigen_array = np.zeros((m.nkx, m.nky, m.nbnd))

    for j in range(m.nky):
        for i in range(m.nkx):
            for banda in range(m.nbnd):
                eigen_array[i, j, banda] = d.eigenvalues[kp, bandsfinal[kp, banda]]
            kp += 1

    return eigen_array

def get_delta_eigen_array_and_fermi(eigen_array: np.ndarray, conduction_band: int) -> Tuple[np.ndarray, np.ndarray]:
    delta_eigen_array = np.zeros((m.nkx, m.nky, conduction_band + 1, conduction_band + 1))
    fermi = np.zeros((m.nkx, m.nky, conduction_band + 1, conduction_band + 1))

    for s in band_list:
        for sprime in band_list:
            delta_eigen_array[:, :, s, sprime] = eigen_array[:, :, s] - eigen_array[:, :, sprime]

            if s <= m.vb < sprime:
                fermi[:, :, s, sprime] = 1
            elif sprime <= m.vb < s:
                fermi[:, :, s, sprime] = -1

    return delta_eigen_array, fermi

def compute_condutivity(omega:float, delta_eigen_array: np.ndarray, fermi: np.ndarray, broadning: complex) -> Tuple[float, np.ndarray]:
    omegaarray = np.full(OMEGA_SHAPE, omega + broadning)
    gamma = CONST * delta_eigen_array / (omegaarray - delta_eigen_array)        # factor that multiplies
    sig = np.full((2, 2), 0.0 + 0j)                                             # matrix sig_xx, sig_xy, sig_yy, sig_yx

    for s in band_list:                                                          # runs through index s
        for sprime in band_list:                                                 # runs through index s'
            if s == sprime:
                continue

            for beta in range(2):                                               # beta is spatial coordinate
                for alpha in range(2):                                          # alpha is spatial coordinate

                    sig[alpha, beta] += np.sum(
                        gamma[:, :, sprime, s]
                        * berry_connections[s][sprime][alpha]
                        * berry_connections[sprime][s][beta]
                        * fermi[:, :, s, sprime]
                    )

    return (omega, sig * VK)

#TODO: ADD assertions to all functions in order to check if the inputs are correct
#IDEA: Maybe create a type checking decorator (USE pydantic)
def run_conductivity(conduction_band: int, npr: int = 1, energy_max: float = 2.5, energy_step: float = 0.001, broadning: complex = 0.01j, logger_name: str = "condutivity", logger_level: int = logging.INFO, flush: bool = False):
    global band_list, berry_connections, OMEGA_SHAPE, CONST, VK
    logger = log(logger_name, "CONDUCTIVITY", level=logger_level, flush=flush)

    logger.header()

    ###########################################################################
    # 1. DEFINING THE CONSTANTS
    ########################################################################### 
    RY    = 13.6056923                                                          # Conversion factor from Ry to eV
    VK    = m.step * m.step / (2 * np.pi) ** 2                                  # element of volume in k-space in units of bohr^-1
    # the '4' comes from spin degeneracy, that is summed in s and s'
    CONST = 4 * 2j / (2 * np.pi) ** 2                                           # = i2e^2/hslash 1/(2pi)^2     in Rydberg units

    band_list   = list(range(conduction_band + 1))

    #TODO: add function docstring with these comments
    # Maximum energy (Ry)
    # Energy step (Ry)
    # energy broading (Ry)

    OMEGA_SHAPE             = (m.nkx, m.nky, conduction_band + 1, conduction_band + 1)
    berry_conn_size  = 2 * m.nkx * m.nky * (conduction_band + 1) ** 2
    berry_conn_shape = (conduction_band + 1, conduction_band + 1, 2, m.nkx, m.nky)
    ###########################################################################
    # 2. STDOUT THE PARAMETERS
    ########################################################################### 
    logger.info(f"\tUsing {npr} processes")

    logger.info(f"\n\tList of bands: {band_list}")
    logger.info(f"\tNumber of k-points in each direction: {m.nkx} {m.nky} {m.nkz}")
    logger.info(f"\tNumber of bands: {m.nbnd}")
    logger.info(f"\tk-points step, dk {m.step}")                                    # Defines the step for gradient calculation dk

    logger.info(f"\n\tMaximum energy (Ry): {energy_max}")
    logger.info(f"\tEnergy step (Ry): {energy_step}")
    logger.info(f"\tEnergy broadning (Ry): {np.imag(broadning)}")
    logger.info(f"\tConstant 4e^2/hslash 1/(2pi)^2 in Rydberg units: {np.imag(CONST)}")
    logger.info(f"\tVolume (area) in k space: {VK}\n")

    ###########################################################################
    # 3. CREATE ALL THE ARRAYS
    ###########################################################################
    bandsfinal               = np.load(os.path.join(m.workdir, "bandsfinal.npy"))
    signalfinal              = np.load(os.path.join(m.workdir, "signalfinal.npy"))                       #NOTE: Not used
    eigen_array              = correct_eigenvalues(bandsfinal)
    berry_connections        = load_berry_connections(conduction_band, berry_conn_size, berry_conn_shape)
    delta_eigen_array, fermi = get_delta_eigen_array_and_fermi(eigen_array, conduction_band)

    ###########################################################################
    # 4. CALCULATE THE CONDUCTIVITY
    ###########################################################################
    with Pool(npr) as pool:
        work_load = product(np.arange(0, energy_max + energy_step, energy_step), [delta_eigen_array], [fermi], [broadning])
        sigma = dict(pool.starmap(compute_condutivity, work_load))

    ###########################################################################
    # 5. SAVE OUTPUT
    ###########################################################################
    with open(os.path.join(m.workdir, "sigmar.dat"), "w") as sigm:
        sigm.write("# Energy (eV), sigma_xx,  sigma_yy,  sigma_yx,  sigma_xy\n")
        for omega in np.arange(0, energy_max + energy_step, energy_step):
            outp = "{0:.4f}  {1:.6f}  {2:.6f}  {3:.6f}  {4:.6f}\n"
            sigm.write(
                outp.format(
                    omega * RY,
                    np.real(sigma[omega][0, 0]),
                    np.real(sigma[omega][1, 1]),
                    np.real(sigma[omega][1, 0]),
                    np.real(sigma[omega][0, 1]),
                )
            )
    logger.info("\tReal part of conductivity saved to file sigmar.dat")

    with open(os.path.join(m.workdir, "sigmai.dat"), "w") as sigm:
        sigm.write("# Energy (eV), sigma_xx,  sigma_yy,  sigma_yx,  sigma_xy\n")
        for omega in np.arange(0, energy_max + energy_step, energy_step):
            outp = "{0:.4f}  {1:.6f}  {2:.6f}  {3:.6f}  {4:.6f}\n"
            sigm.write(
                outp.format(
                    omega * RY,
                    np.imag(sigma[omega][0, 0]),
                    np.imag(sigma[omega][1, 1]),
                    np.imag(sigma[omega][1, 0]),
                    np.imag(sigma[omega][0, 1]),
                )
            )
    logger.info("\tImaginary part of conductivity saved to file sigmai.dat")

    ###########################################################################
    # Finished
    ###########################################################################
    logger.footer()