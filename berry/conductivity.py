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
except:
    pass


def load_berry_connections(conduction_band: int, berry_conn_size: int, berry_conn_shape: Tuple[Sequence[int]]) -> np.ndarray:
    base = Array(ctypes.c_double, berry_conn_size * 2, lock=False)
    berry_connections = np.frombuffer(base, dtype=np.complex128).reshape(berry_conn_shape)

    for i in range(initial_band, conduction_band + 1):
        for j in range(initial_band, conduction_band + 1):
            berry_connections[i - initial_band, j - initial_band] = np.load(os.path.join(m.geometry_dir, f"berryConn{i}_{j}.npy"))

    return berry_connections


def correct_eigenvalues(bandsfinal: np.ndarray) -> np.ndarray:
    kp = 0
    eigenvalues = d.eigenvalues[:,m.initial_band:] # initial band correction
    if m.dimensions == 1:
        eigen_array = np.zeros((m.nkx, number_of_bands))
        for i in range(m.nkx):
            for banda in range(number_of_bands):
                eigen_array[i, banda] = eigenvalues[kp, bandsfinal[kp, banda]]
            kp += 1
    elif m.dimensions == 2:
        eigen_array = np.zeros((m.nkx, m.nky, number_of_bands))
        for j in range(m.nky):
            for i in range(m.nkx):
                for banda in range(number_of_bands):
                    eigen_array[i, j, banda] = eigenvalues[kp, bandsfinal[kp, banda]]
                kp += 1
    else:
        eigen_array = np.zeros((m.nkx, m.nky, m.nkz, number_of_bands))        
        for l in range(m.nkz):
            for j in range(m.nky):
                for i in range(m.nkx):
                    for banda in range(number_of_bands):
                        eigen_array[i, j, l, banda] = eigenvalues[kp, bandsfinal[kp, banda]]
                    kp += 1


    return eigen_array

def get_delta_eigen_array_and_fermi(eigen_array: np.ndarray, conduction_band: int) -> Tuple[np.ndarray, np.ndarray]:
    if m.dimensions == 1:
        delta_eigen_array = np.zeros((m.nkx, conduction_band + 1, conduction_band + 1))
        fermi = np.zeros((m.nkx, conduction_band + 1, conduction_band + 1))
    elif m.dimensions == 2:
        delta_eigen_array = np.zeros((m.nkx, m.nky, conduction_band + 1, conduction_band + 1))
        fermi = np.zeros((m.nkx, m.nky, conduction_band + 1, conduction_band + 1))
    else:
        delta_eigen_array = np.zeros((m.nkx, m.nky, m.nkz, conduction_band + 1, conduction_band + 1))
        fermi = np.zeros((m.nkx, m.nky, m.nkz, conduction_band + 1, conduction_band + 1))

    for s in band_list:
        for sprime in band_list:
            if m.dimensions == 1:
                delta_eigen_array[:, s, sprime] = eigen_array[:, s] - eigen_array[:, sprime]
                if s <= m.vb - initial_band < sprime:
                    fermi[:, s, sprime] = 1
                elif sprime <= m.vb - initial_band < s:
                    fermi[:, s, sprime] = -1
            elif m.dimensions == 2:
                delta_eigen_array[:, :, s, sprime] = eigen_array[:, :, s] - eigen_array[:, :, sprime]
                if s <= m.vb - initial_band < sprime:
                    fermi[:, :, s, sprime] = 1
                elif sprime <= m.vb - initial_band < s:
                    fermi[:, :, s, sprime] = -1
            else:
                delta_eigen_array[:, :, :, s, sprime] = eigen_array[:, :, :, s] - eigen_array[:, :, :, sprime]
                if s <= m.vb - initial_band < sprime:
                    fermi[:, :, :, s, sprime] = 1
                elif sprime <= m.vb - initial_band < s:
                    fermi[:, :, :, s, sprime] = -1
    return delta_eigen_array, fermi

def compute_condutivity(omega:float, delta_eigen_array: np.ndarray, fermi: np.ndarray, broadning: complex) -> Tuple[float, np.ndarray]:
    omegaarray = np.full(OMEGA_SHAPE, omega + broadning)
    gamma = CONST * delta_eigen_array / (omegaarray - delta_eigen_array)        # factor that multiplies
    sig = np.full((m.dimensions, m.dimensions), 0.0 + 0j)                       # matrix sig_xx, sig_xy, sig_yy, sig_yx, etc

    if m.dimensions == 1:
        for s in band_list:                                                         # runs through index s
            for sprime in band_list:                                                # runs through index s'
                if s == sprime:
                    continue
                for beta in range(m.dimensions):                                    # beta is spatial coordinate
                    for alpha in range(m.dimensions):                               # alpha is spatial coordinate

                        sig[alpha, beta] += np.sum(
                            gamma[:, sprime, s]
                            * berry_connections[s][sprime][alpha]
                            * berry_connections[sprime][s][beta]
                            * fermi[:, s, sprime]
                        )
    elif m.dimensions == 2:
        for s in band_list:                                                         # runs through index s
            for sprime in band_list:                                                # runs through index s'
                if s == sprime:
                    continue
                for beta in range(m.dimensions):                                    # beta is spatial coordinate
                    for alpha in range(m.dimensions):                               # alpha is spatial coordinate

                        sig[alpha, beta] += np.sum(
                            gamma[:, :, sprime, s]
                            * berry_connections[s][sprime][alpha]
                            * berry_connections[sprime][s][beta]
                            * fermi[:, :, s, sprime]
                        )
    else:
        for s in band_list:                                                         # runs through index s
            for sprime in band_list:                                                # runs through index s'
                if s == sprime:
                    continue
                for beta in range(m.dimensions):                                    # beta is spatial coordinate
                    for alpha in range(m.dimensions):                               # alpha is spatial coordinate

                        sig[alpha, beta] += np.sum(
                            gamma[:, :, :, sprime, s]
                            * berry_connections[s][sprime][alpha]
                            * berry_connections[sprime][s][beta]
                            * fermi[:, :, :, s, sprime]
                        )

    return (omega, sig * VK)

#TODO: ADD assertions to all functions in order to check if the inputs are correct
#IDEA: Maybe create a type checking decorator (USE pydantic)
def run_conductivity(conduction_band: int, npr: int = 1, energy_max: float = 2.5, energy_step: float = 0.001, brd: float = 0.01, logger_name: str = "condutivity", logger_level: int = logging.INFO, flush: bool = False):
    global band_list, berry_connections, OMEGA_SHAPE, CONST, VK, initial_band, number_of_bands
    logger = log(logger_name, "CONDUCTIVITY", level=logger_level, flush=flush)
    # conduction_band is the number of the highest conduction band to consider.

    logger.header()

    initial_band = m.initial_band if m.initial_band != "dummy" else 0                # for backward compatibility
    number_of_bands = m.number_of_bands if m.number_of_bands != "dummy" else m.nbnd  # for backward compatibility
    broadning = brd*1j
    ###########################################################################
    # 1. DEFINING THE CONSTANTS
    ########################################################################### 
    RY    = 13.6056923                                                          # Conversion factor from Ry to eV
    VK    = (m.step / 2 * np.pi) ** m.dimensions                                # element of volume in k-space in units of bohr^-1
    # the '4' comes from spin degeneracy, that is summed in s and s'
    if m.noncolin:
        CONST = 2j / (2 * np.pi) ** m.dimensions
    else:
        CONST = 4 * 2j / (2 * np.pi) ** m.dimensions                            # = i2e^2/hslash 1/(2pi)^2     in Rydberg units

    band_list   = list(range(conduction_band + 1 - initial_band))

    #TODO: add function docstring with these comments
    # Maximum energy (Ry)
    # Energy step (Ry)
    # energy broading (Ry)

    cb = conduction_band + 1 - initial_band
    if m.dimensions == 1:
        OMEGA_SHAPE = (m.nkx, cb, cb)
        berry_conn_size  = 2 * m.nkx * (cb) ** 2
        berry_conn_shape = (cb, cb, 2, m.nkx)
    elif m.dimensions == 2:
        OMEGA_SHAPE = (m.nkx, m.nky, cb, cb)
        berry_conn_size  = 2 * m.nkx * m.nky * (cb) ** 2
        berry_conn_shape = (cb, cb, 2, m.nkx, m.nky)
    else:
        OMEGA_SHAPE = (m.nkx, m.nky, m.nkz, cb, cb)
        berry_conn_size  = 2 * m.nkx * m.nky * m.nkz * (cb) ** 2
        berry_conn_shape = (cb, cb, 2, m.nkx, m.nky, m.nkz)

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
    if m.noncolin:
        logger.info(f"\tThis is a noncolinear calculation.")
        logger.info(f"\tConstant 2e^2/hslash 1/(2pi)^d in Rydberg units: {np.imag(CONST)}")
    else:
        logger.info(f"\tThis is a no spin calculation.")
        logger.info(f"\tConstant 4e^2/hslash 1/(2pi)^d in Rydberg units: {np.imag(CONST)}")
    logger.info(f"\tNumber of dimensions d = {m.dimensions}")
    logger.info(f"\tVolume (area) in k space: {VK}\n")

    ###########################################################################
    # 3. CREATE ALL THE ARRAYS
    ###########################################################################
    bandsfinal               = np.load(os.path.join(m.data_dir, "bandsfinal.npy"))
    eigen_array              = correct_eigenvalues(bandsfinal)
    berry_connections        = load_berry_connections(conduction_band, berry_conn_size, berry_conn_shape)
    delta_eigen_array, fermi = get_delta_eigen_array_and_fermi(eigen_array, conduction_band - initial_band)

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
        if m.dimensions == 1:
            sigm.write("# Energy (eV), sigma\n")
            for omega in np.arange(0, energy_max + energy_step, energy_step):
                outp = "{0:.4f}  {1:.6f} \n"
                sigm.write(
                    outp.format(
                        omega * RY,
                        np.real(sigma[omega][0, 0])
                    )
                )
        elif m.dimensions == 2:
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
        else:
            sigm.write("# Energy (eV), sigma_xx,  sigma_yy,  sigma_zz,  sigma_xy, sigma_xz,  sigma_yx,  sigma_yz,  sigma_zx,  sigma_zy\n")
            for omega in np.arange(0, energy_max + energy_step, energy_step):
                outp = "{0:.4f}  {1:.6f}  {2:.6f}  {3:.6f}  {4:.6f}  {5:.6f}  {6:.6f}  {7:.6f}  {8:.6f}  {9:.6f}\n"
                sigm.write(
                    outp.format(
                        omega * RY,
                        np.real(sigma[omega][0, 0]),
                        np.real(sigma[omega][1, 1]),
                        np.real(sigma[omega][2, 2]),
                        np.real(sigma[omega][0, 1]),
                        np.real(sigma[omega][0, 2]),
                        np.real(sigma[omega][1, 0]),
                        np.real(sigma[omega][1, 2]),
                        np.real(sigma[omega][2, 0]),
                        np.real(sigma[omega][2, 1])
                    )
                )    
    logger.info("\tReal part of conductivity saved to file sigmar.dat")

    with open(os.path.join(m.workdir, "sigmai.dat"), "w") as sigm:
        if m.dimensions == 1:
            sigm.write("# Energy (eV), sigma\n")
            for omega in np.arange(0, energy_max + energy_step, energy_step):
                outp = "{0:.4f}  {1:.6f} \n"
                sigm.write(
                    outp.format(
                        omega * RY,
                        np.imag(sigma[omega][0, 0])
                    )
                )
        elif m.dimensions == 2:
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
        else:
            sigm.write("# Energy (eV), sigma_xx,  sigma_yy,  sigma_zz,  sigma_xy, sigma_xz,  sigma_yx,  sigma_yz,  sigma_zx,  sigma_zy\n")
            for omega in np.arange(0, energy_max + energy_step, energy_step):
                outp = "{0:.4f}  {1:.6f}  {2:.6f}  {3:.6f}  {4:.6f}  {5:.6f}  {6:.6f}  {7:.6f}  {8:.6f}  {9:.6f}\n"
                sigm.write(
                    outp.format(
                        omega * RY,
                        np.imag(sigma[omega][0, 0]),
                        np.imag(sigma[omega][1, 1]),
                        np.imag(sigma[omega][2, 2]),
                        np.imag(sigma[omega][0, 1]),
                        np.imag(sigma[omega][0, 2]),
                        np.imag(sigma[omega][1, 0]),
                        np.imag(sigma[omega][1, 2]),
                        np.imag(sigma[omega][2, 0]),
                        np.imag(sigma[omega][2, 1])
                    )
                )    
    logger.info("\tImaginary part of conductivity saved to file sigmai.dat")

    ###########################################################################
    # Finished
    ###########################################################################
    logger.footer()
