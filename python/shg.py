"""
# This program calculates the second harmonic generation condutivity
"""

import sys
import time

import itertools
import numpy as np
from findiff import Gradient
import joblib

# This are the subroutines and functions
import contatempo
from headerfooter import header, footer
import loaddata as d
from comutator import comute, comute3, comutederiv

# pylint: disable=C0103
###################################################################################
if __name__ == "__main__":
    header("SHG", d.version, time.asctime())

    STARTTIME = time.time()  # Starts counting time

    if len(sys.argv) < 3:
        print(
            "     ERROR in number of arguments. Has to have two integers.\n \
               If the first is negative, it will only calculate transitions between the too bands."
        )
        sys.exit("Stop")
    elif len(sys.argv) == 3:
        bandfilled = int(sys.argv[1])  # Number of the last filled band at k=0
        bandempty = int(sys.argv[2])  # Number of the last empty band at k=0
        inputfile = ""
    elif len(sys.argv) == 4:
        bandfilled = int(sys.argv[1])  # Number of the last filled band at k=0
        bandempty = int(sys.argv[2])  # Number of the last empty band at k=0
        inputfile = str(sys.argv[3])  # Name of the file where data for the graphic is

    if bandfilled < 0:
        bandfilled = -bandfilled
        bandlist = [bandfilled, bandempty]
        print(
            "     Calculating just transitions from band "
            + str(bandfilled)
            + " to "
            + str(bandempty)
        )
    else:
        bandlist = list(range(bandempty + 1))
        print(
            "     Calculating transitions from bands <"
            + str(bandfilled)
            + " to bands up to "
            + str(bandempty)
        )

    print("     List of bands: ", str(bandlist))

    # Default values:
    enermax = 2.5  # Maximum energy (Ry)
    enerstep = 0.001  # Energy step (Ry)
    broadning = 0.01j  # energy broadning (Ry)
    if inputfile != "":
        with open(inputfile, "r") as le:
            inputvar = le.read().split("\n")
        le.close()
        # Read that from input file
        for i in inputvar:
            ii = i.split()
            if len(ii) == 0:
                continue
            if ii[0] == "enermax":
                enermax = float(ii[1])
            if ii[0] == "enerstep":
                enerstep = float(ii[1])
            if ii[0] == "broadning":
                broadning = 1j * float(ii[1])

    RY = 13.6056923  # Conversion factor from Ry to eV
    ################################################ Read data
    print("     Start reading data")

    # Reading data needed for the run

    print("     Number of k-points in each direction:", d.nkx, d.nky, d.nkz)
    print("     Number of bands:", d.nbnd)
    print("     k-points step, dk", d.step)  # Defines the step for gradient calculation
    print()
    print("     Occupations loaded")  # d.occupations = np.array(nks,d.nbnd)
    print("     Eigenvalues loaded")  # d.eigenvalues = np.array(nks,d.nbnd)
    with open("bandsfinal.npy", "rb") as f:
        bandsfinal = np.load(f)
    f.close()
    print("     bandsfinal.npy loaded")
    with open("signalfinal.npy", "rb") as f:
        signalfinal = np.load(f)
    f.close()
    print("     signalfinal.npy loaded")
    print()

    sys.stdout.flush()
    # sys.exit("Stop")

    berryConnection = {}

    for i, j in itertools.product(range(bandempty + 1), range(bandempty + 1)):
        index = str(i) + " " + str(j)
        filename = "./berryCon" + str(i) + "-" + str(j)
        berryConnection[index] = joblib.load(filename + ".gz")  # Berry connection

    # sys.exit("Stop")
    Earray = np.zeros((d.nkx, d.nky, d.nbnd))  # Eigenvalues corrected for the new bands

    kp = 0
    for j in range(d.nky):  # Energy in Ry
        for i in range(d.nkx):
            for banda in range(d.nbnd):
                Earray[i, j, banda] = d.eigenvalues[kp, bandsfinal[kp, banda]]
            kp += 1
    #        print(Earray[i,j,banda] )

    print("     Finished reading data")
    # sys.exit("Stop")

    ################################################## Finished reading data

    grad = Gradient(h=[d.step, d.step], acc=3)  # Defines gradient function in 2D
    ##################################################

    CONST = (
        2 * np.sqrt(2) * 2 / (2 * np.pi) ** 2
    )  # = -2e^3/hslash 1/(2pi)^2     in Rydberg units
    # the 2e comes from having two electrons per band
    # another minus comes from the negative charge
    vk = d.step * d.step / (2 * np.pi) ** 2  # element of volume in k-space in units of bohr^-1
    # it is actually an area, because we have a 2D crystal
    print("     Maximum energy (Ry): " + str(enermax))
    print("     Energy step (Ry): " + str(enerstep))
    print("     Energy broadning (Ry): " + str(np.imag(broadning)))
    print(
        "     Constant -2e^3/hslash 1/(2pi)^2     in Rydberg units: "
        + str(np.real(CONST))
    )
    print("     Volume (area) in k space: " + str(vk))

    sigma = {}  # Dictionary where the conductivity will be stored
    fermi = np.zeros((d.nkx, d.nky, bandempty + 1, bandempty + 1))
    dE = np.zeros((d.nkx, d.nky, bandempty + 1, bandempty + 1))
    graddE = np.zeros((2, d.nkx, d.nky, bandempty + 1, bandempty + 1), dtype=complex)
    gamma1 = np.zeros((d.nkx, d.nky, bandempty + 1, bandempty + 1), dtype=complex)
    gamma2 = np.zeros((d.nkx, d.nky, bandempty + 1, bandempty + 1), dtype=complex)
    gamma3 = np.zeros((d.nkx, d.nky, bandempty + 1, bandempty + 1), dtype=complex)
    gamma12 = np.zeros((d.nkx, d.nky, bandempty + 1, bandempty + 1), dtype=complex)
    gamma13 = np.zeros((d.nkx, d.nky, bandempty + 1, bandempty + 1), dtype=complex)

    for s, sprime in itertools.product(bandlist, bandlist):
        dE[:, :, s, sprime] = Earray[:, :, s] - Earray[:, :, sprime]
        graddE[:, :, :, s, sprime] = grad(dE[:, :, s, sprime])
        if s <= bandfilled < sprime:
            fermi[:, :, s, sprime] = 1
        elif sprime <= bandfilled < s:
            fermi[:, :, s, sprime] = -1

    #    print(dE[:,:,s,sprime])
    #    print(graddE[:,:,:,s,sprime])
    # e = comute(berryConnection,s,sprime,alpha,beta)

    for omega in np.arange(0, enermax + enerstep, enerstep):
        omegaarray = np.full(
            (d.nkx, d.nky, bandempty + 1, bandempty + 1), omega + broadning
        )  # in Ry
        # matrix sig_xxx,sig_xxy,...,sig_yyx,sig_yyy
        sig = np.full((d.nkx, d.nky, 2, 2, 2), 0.0 + 0j, dtype=complex)

        # factor called dE/g in paper times leading constant
        gamma1 = CONST * dE / (2 * omegaarray - dE)
        # factor f/h^2 in paper (-) to account for change in indices in f and h
        gamma2 = -fermi / np.square(omegaarray - dE)
        # factor f/h in paper (index reference is of h, not f, in equation)
        gamma3 = -fermi / (omegaarray - dE)

        for s, sprime in itertools.product(
            bandlist, bandlist
        ):  # runs through index s, s'
            gamma12[:, :, s, sprime] = gamma1[:, :, s, sprime] * gamma2[:, :, s, sprime]
            gamma13[:, :, s, sprime] = gamma1[:, :, s, sprime] * gamma3[:, :, s, sprime]

        ##   sys.exit("Stop")
        # beta, alpha1, alpha2 are spatial coordinates
        for beta, alpha1, alpha2 in itertools.product(range(2), range(2), range(2)):
            for s, sprime in itertools.product(bandlist, bandlist):  # runs index s,s'
                if s == sprime:
                    continue
                sig[:, :, beta, alpha1, alpha2] += (
                    (
                        graddE[alpha2, :, :, s, sprime]
                        * comute(berryConnection, sprime, s, beta, alpha1)
                        + graddE[alpha1, :, :, s, sprime]
                        * comute(berryConnection, sprime, s, beta, alpha2)
                    )
                    * gamma12[:, :, s, sprime]
                    * 0.5
                )

                sig[:, :, beta, alpha1, alpha2] += (
                    comutederiv(berryConnection, s, sprime, beta, alpha1, alpha2, d.step)
                ) * gamma13[:, :, s, sprime]

                for r in bandlist:  # runs through index r
                    if r in (sprime, s):
                        continue
                    sig[:, :, beta, alpha1, alpha2] += (
                        -0.25j
                        * gamma1[:, :, s, sprime]
                        * (
                            comute3(berryConnection, sprime, s, r, beta, alpha2, alpha1)
                            + comute3(
                                berryConnection, sprime, s, r, beta, alpha1, alpha2
                            )
                        )
                        * gamma3[:, :, r, sprime]
                        - (
                            comute3(berryConnection, sprime, s, r, beta, alpha1, alpha2)
                            + comute3(
                                berryConnection, sprime, s, r, beta, alpha2, alpha1
                            )
                        )
                        * gamma3[:, :, s, r]
                    )

        ##       print(sig)

        sigma[omega] = np.sum(np.sum(sig, axis=0), axis=0) * vk

    ##   sys.exit("Stop")

    with open("sigma2r.dat", "w") as sigm:
        sigm.write(
            "# Energy (eV), sigma_xxx, sigma_yyy, sigma_xxy, sigma_xyx, sigma_xyy, \
                                   sigma_yyx, sigma_yxy, sigma_yxx\n"
        )
        for omega in np.arange(0, enermax + enerstep, enerstep):
            outp = "{0:.4f}  {1:.4e}  {2:.4e}  {3:.4e}  {4:.4e}  {5:.4e}  \
                    {6:.4e}  {7:.4e}  {8:.4e}\n"
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
    sigm.close()
    print("     Real part of SHG saved to file sigma2r.dat")

    with open("sigma2i.dat", "w") as sigm:
        sigm.write(
            "# Energy (eV), sigma_xxx, sigma_yyy, sigma_xxy, sigma_xyx, sigma_xyy, \
                                 sigma_yyx, sigma_yxy, sigma_yxx\n"
        )
        for omega in np.arange(0, enermax + enerstep, enerstep):
            outp = "{0:.4f}  {1:.4e}  {2:.4e}  {3:.4e}  {4:.4e}  {5:.4e}  \
                    {6:.4e}  {7:.4e}  {8:.4e}\n"
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
    sigm.close()
    print("     Imaginary part of SHG saved to file sigma2i.dat")

    # sys.exit("Stop")

    ###################################################################################
    # Finished
    footer(contatempo.tempo(STARTTIME, time.time()))
