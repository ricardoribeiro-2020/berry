"""
  This program calculates the linear conductivity from the Berry connections
"""
from multiprocessing import Array, Pool
from typing import Tuple
from itertools import product

import sys
import time
import ctypes

import numpy as np

from cli import conductivity_cli
from contatempo import time_fn
from headerfooter import header, footer

import contatempo
import loaddata as d


# pylint: disable=C0103
###################################################################################
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
def get_delta_eigen_array_and_fermi(eigen_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    delta_eigen_array = np.zeros((d.nkx, d.nky, BANDEMPTY + 1, BANDEMPTY + 1))
    fermi = np.zeros((d.nkx, d.nky, BANDEMPTY + 1, BANDEMPTY + 1))

    for s in BANDLIST:
        for sprime in BANDLIST:
            delta_eigen_array[:, :, s, sprime] = eigen_array[:, :, s] - eigen_array[:, :, sprime]

            if s <= BANDFILLED < sprime:
                fermi[:, :, s, sprime] = 1
            elif sprime <= BANDFILLED < s:
                fermi[:, :, s, sprime] = -1

    return delta_eigen_array, fermi

@time_fn(0, prefix="\t", display_arg_name=True)
def compute_condutivity(omega:float, delta_eigen_array: np.ndarray, fermi: np.ndarray) -> Tuple[float, np.ndarray]:
    omegaarray = np.full(OMEGA_SHAPE, omega + BROADING)
    gamma = CONST * delta_eigen_array / (omegaarray - delta_eigen_array)        # factor that multiplies
    sig = np.full((2, 2), 0.0 + 0j)                                             # matrix sig_xx, sig_xy, sig_yy, sig_yx

    for s in BANDLIST:                                                          # runs through index s
        for sprime in BANDLIST:                                                 # runs through index s'
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

if __name__ == "__main__":
    args = conductivity_cli()

    header("CONDUTIVITY", d.version, time.asctime())
    STARTTIME = time.time()

    ###########################################################################
    # 1. DEFINING THE CONSTANTS
    ###########################################################################
    RY    = 13.6056923                                                          # Conversion factor from Ry to eV
    VK    = d.step * d.step / (2 * np.pi) ** 2                                  # element of volume in k-space in units of bohr^-1
    # the '4' comes from spin degeneracy, that is summed in s and s'
    CONST = 4 * 2j / (2 * np.pi) ** 2                                           # = i2e^2/hslash 1/(2pi)^2     in Rydberg units

    BANDFILLED = 3
    BANDEMPTY  = args["BANDEMPTY"]
    BANDLIST   = list(range(BANDEMPTY + 1))

    NPR = args["NPR"]
    ENERMAX  = args["ENERMAX"]                                                  # Maximum energy (Ry)
    ENERSTEP = args["ENERSTEP"]                                                 # Energy step (Ry)
    BROADING = args["BROADNING"]                                                # energy broading (Ry)

    OMEGA_SHAPE             = (d.nkx, d.nky, BANDEMPTY + 1, BANDEMPTY + 1)
    BERRY_CONNECTIONS_SIZE  = 2 * d.nkx * d.nky * (BANDEMPTY + 1) ** 2
    BERRY_CONNECTIONS_SHAPE = (BANDEMPTY + 1, BANDEMPTY + 1, 2, d.nkx, d.nky)
    ###########################################################################
    # 2. STDOUT THE PARAMETERS
    ###########################################################################
    print(f"\tList of bands: {BANDLIST}")
    print(f"\tNumber of k-points in each direction: {d.nkx} {d.nky} {d.nkz}")
    print(f"\tNumber of bands: {d.nbnd}")
    print(f"\tk-points step, dk {d.step}")                                      # Defines the step for gradient calculation dk

    print(f"\n\tMaximum energy (Ry): {ENERMAX}")
    print(f"\tEnergy step (Ry): {ENERSTEP}")
    print(f"\tEnergy BROADING (Ry): {np.imag(BROADING)}")
    print(f"\tConstant 4e^2/hslash 1/(2pi)^2 in Rydberg units: {np.imag(CONST)}")
    print(f"\tVolume (area) in k space: {VK}\n")
    sys.stdout.flush()
    ###########################################################################
    # 3. CREATE ALL THE ARRAYS
    ###########################################################################
    bandsfinal               = np.load("bandsfinal.npy")
    signalfinal              = np.load("signalfinal.npy")                       # Not used
    eigen_array              = correct_eigenvalues(bandsfinal)
    berry_connections        = load_berry_connections()
    delta_eigen_array, fermi = get_delta_eigen_array_and_fermi(eigen_array)
    ###########################################################################
    # 4. CALCULATE THE CONDUCTIVITY
    ###########################################################################
    with Pool(NPR) as pool:
        sigma = dict(pool.starmap(compute_condutivity, product(np.arange(0, ENERMAX + ENERSTEP, ENERSTEP), [delta_eigen_array], [fermi])))
    ###########################################################################
    # 5. SAVE OUTPUT
    ###########################################################################
    with open("sigmar.dat", "w") as sigm:
        sigm.write("# Energy (eV), sigma_xx,  sigma_yy,  sigma_yx,  sigma_xy\n")
        for omega in np.arange(0, ENERMAX + ENERSTEP, ENERSTEP):
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
    print("\n\tReal part of conductivity saved to file sigmar.dat")

    with open("sigmai.dat", "w") as sigm:
        sigm.write("# Energy (eV), sigma_xx,  sigma_yy,  sigma_yx,  sigma_xy\n")
        for omega in np.arange(0, ENERMAX + ENERSTEP, ENERSTEP):
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
    print("\tImaginary part of conductivity saved to file sigmai.dat")
    ###################################################################################
    # Finished
    ###################################################################################
    footer(contatempo.tempo(STARTTIME, time.time()))
