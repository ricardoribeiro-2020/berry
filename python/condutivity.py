"""
  This program calculates the linear conductivity from the Berry connections
"""

import sys
import time

import numpy as np

# This are the subroutines and functions
import contatempo
from headerfooter import header, footer
import loaddata as d

# pylint: disable=C0103
###################################################################################
if __name__ == "__main__":
    header("CONDUTIVITY", d.version, time.asctime())

    STARTTIME = time.time()  # Starts counting time

    if len(sys.argv) < 3:
        print(
            " \tERROR in number of arguments. Has to have two integers.\n \
              If the first is negative, it will only calculate transitions between the too bands."
        )
        sys.exit("Stop")
    elif len(sys.argv) == 3:
        bandfilled = d.vb  # Number of the last filled band at k=0
        bandempty = int(sys.argv[2])  # Number of the last empty band at k=0
        inputfile = ""
    elif len(sys.argv) == 4:
        bandfilled = d.vb  # Number of the last filled band at k=0
        bandempty = int(sys.argv[2])  # Number of the last empty band at k=0
        inputfile = str(sys.argv[3])  # Name of the file where data for the graphic is

    if bandfilled < 0:
        bandfilled = -bandfilled
        bandlist = [bandfilled, bandempty]
        print(
            " \tCalculating just transitions from band "
            + str(bandfilled)
            + " to "
            + str(bandempty)
        )
    else:
        bandlist = list(range(bandempty + 1))
        print(
            " \tCalculating transitions from bands <"
            + str(bandfilled)
            + " to bands up to "
            + str(bandempty)
        )

    print("\tList of bands: ", str(bandlist))

    # Default values:
    enermax = 2.5  # Maximum energy (Ry)
    enerstep = 0.001  # Energy step (Ry)
    broadning = 0.01j  # energy broadning (Ry)
    if inputfile != "":
        with open(inputfile, "r") as le:
            inputvar = le.read().split("\n")
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
    print("\tStart reading data")

    # Reading data needed for the run
    print("\tNumber of k-points in each direction:", d.nkx, d.nky, d.nkz)
    print("\tNumber of bands:", d.nbnd)
    print("\tk-points step, dk", d.step)  # Defines the step for gradient calculation dk
    print()
    print("\tOccupations loaded")  # d.occupations = np.array(nks,d.nbnd)
    print("\tEigenvalues loaded")  # d.eigenvalues = np.array(nks,d.nbnd)
    with open("bandsfinal.npy", "rb") as fich:
        bandsfinal = np.load(fich)
    fich.close()
    print("\tbandsfinal.npy loaded")
    with open("signalfinal.npy", "rb") as fich:
        signalfinal = np.load(fich)
    fich.close()
    print("\tsignalfinal.npy loaded")
    print()

    sys.stdout.flush()
    # sys.exit("Stop")

    berry_connection = {}
    for i in range(bandempty + 1):
        for j in range(bandempty + 1):
            index = str(i) + " " + str(j)
            filename = "./berryConn" + str(i) + "_" + str(j)
            berry_connection[index] = np.load(filename + ".npy")  # Berry connection

    # sys.exit("Stop")
    Earray = np.zeros((d.nkx, d.nky, d.nbnd))  # Eigenvalues corrected for the new bands

    kp = 0
    for j in range(d.nky):  # Energy in Ry
        for i in range(d.nkx):
            for banda in range(d.nbnd):
                Earray[i, j, banda] = d.eigenvalues[kp, bandsfinal[kp, banda]]
            kp += 1
    #      print(Earray[i,j,banda] )

    print("\tFinished reading data")
    # sys.exit("Stop")

    ################################################## Finished reading data

    CONST = 4 * 2j / (2 * np.pi) ** 2  # = i2e^2/hslash 1/(2pi)^2     in Rydberg units
    # the '4' comes from spin degeneracy, that is summed in s and s'
    VK = d.step * d.step / (2 * np.pi) ** 2  # element of volume in k-space in units of bohr^-1

    print("\tMaximum energy (Ry): " + str(enermax))
    print("\tEnergy step (Ry): " + str(enerstep))
    print("\tEnergy broadning (Ry): " + str(np.imag(broadning)))
    print(
        " \tConstant 4e^2/hslash 1/(2pi)^2     in Rydberg units: "
        + str(np.imag(CONST))
    )
    print("\tVolume (area) in k space: " + str(VK))

    sigma = {}  # Dictionary where the conductivity will be stored
    FERMI = np.zeros((d.nkx, d.nky, bandempty + 1, bandempty + 1))
    dE = np.zeros((d.nkx, d.nky, bandempty + 1, bandempty + 1))

    for s in bandlist:
        for sprime in bandlist:
            dE[:, :, s, sprime] = Earray[:, :, s] - Earray[:, :, sprime]

            if s <= bandfilled < sprime:
                FERMI[:, :, s, sprime] = 1
            elif sprime <= bandfilled < s:
                FERMI[:, :, s, sprime] = -1

    # sys.exit("Stop")
    for omega in np.arange(0, enermax + enerstep, enerstep):
        omegaarray = np.full(
            (d.nkx, d.nky, bandempty + 1, bandempty + 1), omega + broadning
        )  # in Ry
        sig = np.full((2, 2), 0.0 + 0j)  # matrix sig_xx,sig_xy,sig_yy,sig_yx

        gamma = CONST * dE / (omegaarray - dE)  # factor that multiplies

        for s in bandlist:  # runs through index s
            for sprime in bandlist:  # runs through index s'
                if s == sprime:
                    continue
                s_sprime = str(s) + " " + str(sprime)
                sprime_s = str(sprime) + " " + str(s)

                for beta in range(2):  # beta is spatial coordinate
                    for alpha in range(2):  # alpha is spatial coordinate

                        sig[alpha, beta] += np.sum(
                            gamma[:, :, sprime, s]
                            * berry_connection[s_sprime][alpha]
                            * berry_connection[sprime_s][beta]
                            * FERMI[:, :, s, sprime]
                        )

        sigma[omega] = sig * VK

    with open("sigmar.dat", "w") as sigm:
        sigm.write("# Energy (eV), sigma_xx,  sigma_yy,  sigma_yx,  sigma_xy\n")
        for omega in np.arange(0, enermax + enerstep, enerstep):
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
    print("\tReal part of conductivity saved to file sigmar.dat")

    with open("sigmai.dat", "w") as sigm:
        sigm.write("# Energy (eV), sigma_xx,  sigma_yy,  sigma_yx,  sigma_xy\n")
        for omega in np.arange(0, enermax + enerstep, enerstep):
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
    footer(contatempo.tempo(STARTTIME, time.time()))
