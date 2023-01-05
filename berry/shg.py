from multiprocessing import Pool, Array
from itertools import product
from typing import Tuple

import os
import ctypes
import logging

from findiff import Gradient

import numpy as np

from berry import log
from berry._subroutines.comutator import comute, comute3, comutederiv

try:
    import berry._subroutines.loaddata as d
    import berry._subroutines.loadmeta as m
except:
    pass


def load_berry_connections(conduction_band: int, berry_conn_size: int, berry_conn_shape: Tuple[int]) -> np.ndarray:
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


def get_fermi_delta_ea_grad_ea(grad: Gradient, eigen_array: np.ndarray, conduction_band: int, ) -> Tuple[np.ndarray, np.ndarray]:
    grad_dea = np.zeros((2, m.nkx, m.nky, conduction_band + 1, conduction_band + 1), dtype=np.complex128)
    delta_ea = np.zeros((m.nkx, m.nky, conduction_band + 1, conduction_band + 1))
    fermi = np.zeros((m.nkx, m.nky, conduction_band + 1, conduction_band + 1))

    for s, sprime in product(band_list, repeat=2):
        delta_ea[:, :, s, sprime] = eigen_array[:, :, s] - eigen_array[:, :, sprime]
        grad_dea[:, :, :, s, sprime] = grad(delta_ea[:, :, s, sprime])
        if s <= m.vb < sprime:
            fermi[:, :, s, sprime] = 1
        elif sprime <= m.vb < s:
            fermi[:, :, s, sprime] = -1

    return fermi, delta_ea, grad_dea

def calculate_shg(omega: float, broadning: float):
    omega_array = np.full(OMEGA_SHAPE, omega + broadning)                        # in Ry
    sig = np.full((m.nkx, m.nky, 2, 2, 2), 0, dtype=np.complex128)               # matrix sig_xxx,sig_xxy,...,sig_yyx,sig_yyy

    gamma1 = CONST * delta_ea / (2 * omega_array - delta_ea)                     # factor called dE/g in paper times leading constant
    gamma2 = -fermi / np.square(omega_array - delta_ea)                          # factor f/h^2 in paper (-) to account for change in indices in f and h
    gamma3 = -fermi / (omega_array - delta_ea)                                   # factor f/h in paper (index reference is of h, not f, in equation)

    for s, sprime in product(band_list, repeat=2):                                # runs through index s, s'
        gamma12[:, :, s, sprime] = gamma1[:, :, s, sprime] * gamma2[:, :, s, sprime]
        gamma13[:, :, s, sprime] = gamma1[:, :, s, sprime] * gamma3[:, :, s, sprime]

    # beta, alpha1, alpha2 are spatial coordinates
    for beta, alpha1, alpha2 in product(range(2), range(2), range(2)):
        for s, sprime in product(band_list, band_list):                            # runs through index s, s'
            if s == sprime:
                continue
            sig[:, :, beta, alpha1, alpha2] += (
                (
                    grad_dea[alpha2, :, :, s, sprime]
                    * comute(berry_connections, sprime, s, beta, alpha1)
                    + grad_dea[alpha1, :, :, s, sprime]
                    * comute(berry_connections, sprime, s, beta, alpha2)
                )
                * gamma12[:, :, s, sprime]
                * 0.5
            )

            sig[:, :, beta, alpha1, alpha2] += (comutederiv(berry_connections, s, sprime, beta, alpha1, alpha2, m.step)) * gamma13[:, :, s, sprime]

            for r in band_list:                                                   # runs through index r
                if r in (sprime, s):
                    continue
                sig[:, :, beta, alpha1, alpha2] += (
                    -0.25j * gamma1[:, :, s, sprime] * 
                    (
                        comute3(berry_connections, sprime, s, r, beta, alpha2, alpha1) + 
                        comute3(berry_connections, sprime, s, r, beta, alpha1, alpha2)
                    ) * 
                    gamma3[:, :, r, sprime] - 
                    (
                        comute3(berry_connections, sprime, s, r, beta, alpha1, alpha2) +
                        comute3(berry_connections, sprime, s, r, beta, alpha2, alpha1)
                    ) * 
                    gamma3[:, :, s, r]
                )

    return (omega, np.sum(np.sum(sig, axis=0), axis=0) * VK)


def run_shg(conduction_band: int, npr: int = 1, energy_max: float = 2.5, energy_step: float = 0.001, broadning: complex = 0.01j, logger_name: str = "shg", logger_level: int = logging.INFO, flush: bool = False):
    global gamma1, gamma2, gamma3, gamma12, gamma13, fermi, delta_ea, grad_dea, band_list, berry_connections, OMEGA_SHAPE, CONST, VK
    logger = log(logger_name, "SECOND HARMONIC GENERATOR", level=logger_level, flush=flush)

    logger.header()

    ###########################################################################
    # 1. DEFINING THE CONSTANTS
    ###########################################################################
    RY    = 13.6056923                                                          # Conversion factor from Ry to eV
    VK    = m.step * m.step / (2 * np.pi) ** 2                                  # element of volume in k-space in units of bohr^-1
                                                                                # it is actually an area, because we have a 2D crystal
    CONST = 2 * np.sqrt(2) * 2 / (2 * np.pi) ** 2                               # = -2e^3/hslash 1/(2pi)^2     in Rydberg units
                                                                                # the 2e comes from having two electrons per band
                                                                                # another minus comes from the negative charge

    band_list   = list(range(conduction_band + 1))

    #TODO: Add docstring with these comments
    # Maximum energy (Ry)
    # Energy step (Ry)
    # energy broading (Ry)

    GAMMA_SHAPE             = (m.nkx, m.nky, conduction_band + 1, conduction_band + 1)
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
    logger.info(f"\tk-points step, dk {m.step}")                                      # Defines the step for gradient calculation dk

    logger.info(f"\n\tMaximum energy (Ry): {energy_max}")
    logger.info(f"\tEnergy step (Ry): {energy_step}")
    logger.info(f"\tEnergy broadning (Ry): {np.imag(broadning)}")
    logger.info(f"\tConstant 4e^2/hslash 1/(2pi)^2 in Rydberg units: {np.imag(CONST)}")
    logger.info(f"\tVolume (area) in k space: {VK}\n")

    ###########################################################################
    # 3. CREATE ALL THE ARRAYS
    ###########################################################################
    grad                        = Gradient(h=[m.step, m.step], acc=2)           # Defines gradient function in 2D
    bandsfinal                  = np.load(os.path.join(m.workdir, "bandsfinal.npy"))
    eigen_array                 = correct_eigenvalues(bandsfinal)
    berry_connections           = load_berry_connections(conduction_band, berry_conn_size, berry_conn_shape)
    fermi, delta_ea, grad_dea   = get_fermi_delta_ea_grad_ea(grad, eigen_array, conduction_band)

    gamma1                      = np.zeros(GAMMA_SHAPE, dtype=np.complex128)
    gamma2                      = np.zeros(GAMMA_SHAPE, dtype=np.complex128)
    gamma3                      = np.zeros(GAMMA_SHAPE, dtype=np.complex128)
    gamma12                     = np.zeros(GAMMA_SHAPE, dtype=np.complex128)
    gamma13                     = np.zeros(GAMMA_SHAPE, dtype=np.complex128)

    ###########################################################################
    # 4. SECONG HARMONIC GENERATION
    ###########################################################################
    sigma = {}
    with Pool(npr) as pool:
        results = pool.starmap(calculate_shg, ((omega, broadning) for omega in np.arange(0, energy_max + energy_step, energy_step)))
        for omega, result in results:
            sigma[omega] = result

    ###########################################################################
    # 5. SAVE OUTPUT
    ###########################################################################
    with open(os.path.join(m.workdir, "sigma2r.dat"), "w") as sigm:
        sigm.write("# Energy (eV), sigma_xxx, sigma_yyy, sigma_xxy, sigma_xyx, sigma_xyy, sigma_yyx, sigma_yxy, sigma_yxx\n")
        for omega in np.arange(0, energy_max + energy_step, energy_step):
            outp = "{0:.4f}  {1:.4e}  {2:.4e}  {3:.4e}  {4:.4e}  {5:.4e}  {6:.4e}  {7:.4e}  {8:.4e}\n"
            sigm.write(
                outp.format(
                    omega * RY,
                    np.real(sigma[omega][0, 0, 0]),
                    np.real(sigma[omega][1, 1, 1]),
                    np.real(sigma[omega][0, 0, 1]),
                    np.real(sigma[omega][0, 1, 0]),
                    np.real(sigma[omega][0, 1, 1]),
                    np.real(sigma[omega][1, 1, 0]),
                    np.real(sigma[omega][1, 0, 1]),
                    np.real(sigma[omega][1, 0, 0]),
                )
            )
    logger.info("\tReal part of SHG saved to file sigma2r.dat")

    with open(os.path.join(m.workdir, "sigma2i.dat"), "w") as sigm:
        sigm.write("# Energy (eV), sigma_xxx, sigma_yyy, sigma_xxy, sigma_xyx, sigma_xyy, sigma_yyx, sigma_yxy, sigma_yxx\n")
        for omega in np.arange(0, energy_max + energy_step, energy_step):
            outp = "{0:.4f}  {1:.4e}  {2:.4e}  {3:.4e}  {4:.4e}  {5:.4e}  {6:.4e}  {7:.4e}  {8:.4e}\n"
            sigm.write(
                outp.format(
                    omega * RY,
                    np.imag(sigma[omega][0, 0, 0]),
                    np.imag(sigma[omega][1, 1, 1]),
                    np.imag(sigma[omega][0, 0, 1]),
                    np.imag(sigma[omega][0, 1, 0]),
                    np.imag(sigma[omega][0, 1, 1]),
                    np.imag(sigma[omega][1, 1, 0]),
                    np.imag(sigma[omega][1, 0, 1]),
                    np.imag(sigma[omega][1, 0, 0]),
                )
            )
    logger.info("\tImaginary part of SHG saved to file sigma2i.dat")

    ###################################################################################
    # Finished
    ###################################################################################

    logger.footer()


if __name__ == "__main__":
    run_shg(9, log("shg", "SECOND HARMONIC GENERATION", "version", logging.DEBUG), npr=40)