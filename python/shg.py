"""
# This program calculates the second harmonic generation condutivity
"""
from multiprocessing import Pool, Array
from itertools import product
from typing import Tuple

import sys
import time

from findiff import Gradient

import ctypes
import numpy as np

from cli import shg_cli
from contatempo import time_fn
from comutator import comute, comute3, comutederiv
from log_libs import log

import loaddata as d

LOG: log = log("shg", "SECOND HARMONIC GENERATION", d.version)

# pylint: disable=C0103
###############################################################################
@time_fn(prefix="\t")
def load_berry_connections() -> np.ndarray:
    base = Array(ctypes.c_double, BERRY_CONNECTIONS_SIZE * 2, lock=False)
    berry_connections = np.frombuffer(base, dtype=np.complex128).reshape(BERRY_CONNECTIONS_SHAPE)

    for i in range(BANDEMPTY + 1):
        for j in range(BANDEMPTY + 1):
            berry_connections[i, j] = np.load(f"berryConn{i}_{j}.npy")

    return berry_connections

@time_fn(prefix="\t")
def correct_eigenvalues(bandsfinal: np.ndarray) -> np.ndarray:
    kp = 0
    eigen_array = np.zeros((d.nkx, d.nky, d.nbnd))

    for j in range(d.nky):
        for i in range(d.nkx):
            for banda in range(d.nbnd):
                eigen_array[i, j, banda] = d.eigenvalues[kp, bandsfinal[kp, banda]]
            kp += 1

    return eigen_array

@time_fn(prefix="\t")
def get_fermi_delta_ea_grad_ea(grad: Gradient, eigen_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    grad_dea = np.zeros((2, d.nkx, d.nky, BANDEMPTY + 1, BANDEMPTY + 1), dtype=np.complex128)
    delta_ea = np.zeros((d.nkx, d.nky, BANDEMPTY + 1, BANDEMPTY + 1))
    fermi = np.zeros((d.nkx, d.nky, BANDEMPTY + 1, BANDEMPTY + 1))

    for s, sprime in product(BANDLIST, repeat=2):
        delta_ea[:, :, s, sprime] = eigen_array[:, :, s] - eigen_array[:, :, sprime]
        grad_dea[:, :, :, s, sprime] = grad(delta_ea[:, :, s, sprime])
        if s <= BANDFILLED < sprime:
            fermi[:, :, s, sprime] = 1
        elif sprime <= BANDFILLED < s:
            fermi[:, :, s, sprime] = -1

    return fermi, delta_ea, grad_dea

def calculate_shg(omega):
    omega_array = np.full(OMEGA_SHAPE, omega + BROADNING)                        # in Ry
    sig = np.full((d.nkx, d.nky, 2, 2, 2), 0, dtype=np.complex128)               # matrix sig_xxx,sig_xxy,...,sig_yyx,sig_yyy

    gamma1 = CONST * delta_ea / (2 * omega_array - delta_ea)                     # factor called dE/g in paper times leading constant
    gamma2 = -fermi / np.square(omega_array - delta_ea)                          # factor f/h^2 in paper (-) to account for change in indices in f and h
    gamma3 = -fermi / (omega_array - delta_ea)                                   # factor f/h in paper (index reference is of h, not f, in equation)

    for s, sprime in product(BANDLIST, repeat=2):                                # runs through index s, s'
        gamma12[:, :, s, sprime] = gamma1[:, :, s, sprime] * gamma2[:, :, s, sprime]
        gamma13[:, :, s, sprime] = gamma1[:, :, s, sprime] * gamma3[:, :, s, sprime]

    # beta, alpha1, alpha2 are spatial coordinates
    for beta, alpha1, alpha2 in product(range(2), range(2), range(2)):
        for s, sprime in product(BANDLIST, BANDLIST):                            # runs through index s, s'
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

            sig[:, :, beta, alpha1, alpha2] += (comutederiv(berry_connections, s, sprime, beta, alpha1, alpha2, d.step)) * gamma13[:, :, s, sprime]

            for r in BANDLIST:                                                   # runs through index r
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

if __name__ == "__main__":
    args = shg_cli()

    LOG.header()

    ###########################################################################
    # 1. DEFINING THE CONSTANTS
    ###########################################################################
    RY    = 13.6056923                                                          # Conversion factor from Ry to eV
    VK    = d.step * d.step / (2 * np.pi) ** 2                                  # element of volume in k-space in units of bohr^-1
                                                                                # it is actually an area, because we have a 2D crystal
    CONST = 2 * np.sqrt(2) * 2 / (2 * np.pi) ** 2                               # = -2e^3/hslash 1/(2pi)^2     in Rydberg units
                                                                                # the 2e comes from having two electrons per band
                                                                                # another minus comes from the negative charge

    BANDFILLED = d.vb
    BANDEMPTY  = args["BANDEMPTY"]
    BANDLIST   = list(range(BANDEMPTY + 1))

    NPR = args["NPR"]
    ENERMAX  = args["ENERMAX"]                                                  # Maximum energy (Ry)
    ENERSTEP = args["ENERSTEP"]                                                 # Energy step (Ry)
    BROADNING = args["BROADNING"]                                               # energy broading (Ry)

    GAMMA_SHAPE             = (d.nkx, d.nky, BANDEMPTY + 1, BANDEMPTY + 1)
    OMEGA_SHAPE             = (d.nkx, d.nky, BANDEMPTY + 1, BANDEMPTY + 1)
    BERRY_CONNECTIONS_SIZE  = 2 * d.nkx * d.nky * (BANDEMPTY + 1) ** 2
    BERRY_CONNECTIONS_SHAPE = (BANDEMPTY + 1, BANDEMPTY + 1, 2, d.nkx, d.nky)

    ###########################################################################
    # 2. STDOUT THE PARAMETERS
    ###########################################################################
    LOG.info(f"\tList of bands: {BANDLIST}")
    LOG.info(f"\tNumber of k-points in each direction: {d.nkx} {d.nky} {d.nkz}")
    LOG.info(f"\tNumber of bands: {d.nbnd}")
    LOG.info(f"\tk-points step, dk {d.step}")                                      # Defines the step for gradient calculation dk

    LOG.info(f"\n\tMaximum energy (Ry): {ENERMAX}")
    LOG.info(f"\tEnergy step (Ry): {ENERSTEP}")
    LOG.info(f"\tEnergy broadning (Ry): {np.imag(BROADNING)}")
    LOG.info(f"\tConstant 4e^2/hslash 1/(2pi)^2 in Rydberg units: {np.imag(CONST)}")
    LOG.info(f"\tVolume (area) in k space: {VK}\n")
    sys.stdout.flush()

    ###########################################################################
    # 3. CREATE ALL THE ARRAYS
    ###########################################################################
    grad                        = Gradient(h=[d.step, d.step], acc=2)           # Defines gradient function in 2D
    bandsfinal                  = np.load("bandsfinal.npy")
    signalfinal                 = np.load("signalfinal.npy")                       
    eigen_array                 = correct_eigenvalues(bandsfinal)
    berry_connections           = load_berry_connections()
    fermi, delta_ea, grad_dea   = get_fermi_delta_ea_grad_ea(grad, eigen_array)

    gamma1                      = np.zeros(GAMMA_SHAPE, dtype=np.complex128)
    gamma2                      = np.zeros(GAMMA_SHAPE, dtype=np.complex128)
    gamma3                      = np.zeros(GAMMA_SHAPE, dtype=np.complex128)
    gamma12                     = np.zeros(GAMMA_SHAPE, dtype=np.complex128)
    gamma13                     = np.zeros(GAMMA_SHAPE, dtype=np.complex128)

    ###########################################################################
    # 4. SECONG HARMONIC GENERATION
    ###########################################################################
    sigma = {}
    with Pool(NPR) as pool:
        results = pool.map(calculate_shg, (omega for omega in np.arange(0, ENERMAX + ENERSTEP, ENERSTEP)))
        for omega, result in results:
            sigma[omega] = result

    ###########################################################################
    # 5. SAVE OUTPUT
    ###########################################################################
    with open("sigma2r.dat", "w") as sigm:
        sigm.write("# Energy (eV), sigma_xxx, sigma_yyy, sigma_xxy, sigma_xyx, sigma_xyy, sigma_yyx, sigma_yxy, sigma_yxx\n")
        for omega in np.arange(0, ENERMAX + ENERSTEP, ENERSTEP):
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
    LOG.info("\n\tReal part of SHG saved to file sigma2r.dat")

    with open("sigma2i.dat", "w") as sigm:
        sigm.write("# Energy (eV), sigma_xxx, sigma_yyy, sigma_xxy, sigma_xyx, sigma_xyy, sigma_yyx, sigma_yxy, sigma_yxx\n")
        for omega in np.arange(0, ENERMAX + ENERSTEP, ENERSTEP):
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
    LOG.info("\tImaginary part of SHG saved to file sigma2i.dat")

    ###################################################################################
    # Finished
    ###################################################################################

    LOG.footer()
