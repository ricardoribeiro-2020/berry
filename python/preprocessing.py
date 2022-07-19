"""This program runs a set of dft calculations, and prepares for further processing

    input file has to have these minimum parameters:
        origin of k-points - k0
        number of k-points in each direction - nkx, nky, nkz
        step of the k-points - step
        number of bands - nbnd
    and can have these other, that have defaults:
        number of processors - npr = 1
        dft directory - DFTDIRECTORY = './dft/'
        name_scf = 'scf.in'
        wfcdirectory = './wfc/'
        used to define the point in real space where all phases match: point = 1.178097
        program = 'QE'
        prefix = ''
        outdir = ''
       refname = date and time: it is better to keep the default
"""
__version__ = "v0.3"

import os
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np

import contatempo
import dft
from headerfooter import header, footer
from parserQE import parser
from write_k_points import list_kpoints

# pylint: disable=C0103
###################################################################################
if __name__ == "__main__":
    header("PREPROCESSING", __version__, time.asctime())

    STARTTIME = time.time()  # Starts counting time

    if len(sys.argv) != 2:
        print(" *!*!*!*!*!*!*!*!*!*!*!")
        print(" ERROR in number of arguments. No input file.")
        print(" *!*!*!*!*!*!*!*!*!*!*!")
        print()
        sys.exit("Stop")

    print("     Reading from input file:", sys.argv[1])
    print()

    # Check python version
    python_version = (
                   str(sys.version_info[0])
                   + '.'
                   + str(sys.version_info[1])
                   + '.'
                   + str(sys.version_info[2])
    )
    if not sys.version_info > (2, 7):
        print(" *!*!*!*!*!*!*!*!*!*!*!")
        print(" ERROR: this runs in python3. You are using python2.")
        print(" *!*!*!*!*!*!*!*!*!*!*!")
        print()
        sys.exit("Stop")
    elif not sys.version_info >= (3, 8):
        print("     This program was tested in python 3.8")
        print("     You are using version", python_version, "but it should work")
    else:
        print("     Using python", python_version)
    print()

    # Defaults:
    NPR = 1
    DFTDIRECTORY = "dft/"
    NAMESCF = "scf.in"
    WFCDIRECTORY = "wfc/"
    POINT = 1.178097
    WORKDIR = os.getcwd() + "/"  # Directory where results will show
    PROGRAM = "QE"  # Default dft program is Quantum Espresso
    PREFIX = ""  # Prefix of the DFT calculation (QE)
    OUTDIR = ""  # Output directory of the DFT calculation (QE)
    refname = time.strftime("%d-%m-%Y_%H:%M:%S", time.gmtime())  #unique ref.

    # Open input file for the run
    with open(sys.argv[1], "r") as inputfile:
        INPUTVAR = inputfile.read().split("\n")
    inputfile.close()

    # Read that from input file
    for i in INPUTVAR:
        ii = i.split()
        if len(ii) == 0:
            continue
        if ii[0] == "k0":
            K0 = [float(ii[1]), float(ii[2]), float(ii[3])]
        if ii[0] == "nkx":
            nkx = int(ii[1])
        if ii[0] == "nky":
            nky = int(ii[1])
        if ii[0] == "nkz":
            nkz = int(ii[1])
        if ii[0] == "step":
            step = float(ii[1])
        if ii[0] == "nbnd":
            NBND = int(ii[1])
        if ii[0] == "npr":
            NPR = int(ii[1])
        if ii[0] == "dftdirectory":
            DFTDIRECTORY = ii[1]
        if ii[0] == "name_scf":
            NAMESCF = ii[1]
        if ii[0] == "wfcdirectory":
            WFCDIRECTORY = ii[1]
        if ii[0] == "point":
            POINT = float(ii[1])
        if ii[0] == "program":
            PROGRAM = str(ii[1])
        if ii[0] == "prefix":
            PREFIX = ii[1]
        if ii[0] == "outdir":
            OUTDIR = ii[1]
        if ii[0] == "refname":
            OUTDIR = ii[1]

    # create absolute paths for directories and verify values
    if DFTDIRECTORY[:2] == "./":
        DFTDIRECTORY = WORKDIR + DFTDIRECTORY[2:]
    elif DFTDIRECTORY[:1] != "/":
        DFTDIRECTORY = WORKDIR + DFTDIRECTORY

    OUTDIR = parser("outdir", DFTDIRECTORY + NAMESCF)
    if OUTDIR[:2] == "./":
        OUTDIR = DFTDIRECTORY + OUTDIR[2:]
    elif OUTDIR[:1] != "/":
        OUTDIR = DFTDIRECTORY + OUTDIR

    if WFCDIRECTORY[:2] == "./":
        WFCDIRECTORY = WORKDIR + WFCDIRECTORY[2:]
    elif WFCDIRECTORY[:1] != "/":
        WFCDIRECTORY = WORKDIR + WFCDIRECTORY

    PSEUDODIR = parser("pseudo_dir", DFTDIRECTORY + NAMESCF)
    if PSEUDODIR[:2] == "./":
        PSEUDODIR = DFTDIRECTORY + PSEUDODIR[2:]
    elif PSEUDODIR[:1] != "/":
        PSEUDODIR = DFTDIRECTORY + PSEUDODIR

    NAMENSCF = "n" + NAMESCF
    if NAMENSCF[-3:] != ".in":
        NAMENSCF = NAMENSCF + ".in"
    if PREFIX == "":
        PREFIX = parser("prefix", DFTDIRECTORY + NAMESCF)

    print("     Unique reference of run:", refname)
    print("     Starting k-point of the mesh K0:", str(K0))
    print("     Number of points in the mesh ", str(nkx), str(nky), str(nkz))
    print("     Step of the k-point mesh ", str(step))
    print("     To calculate point in real space where all phases match ", str(POINT))
    print("     Number of bands in the nscf calculation nbnd:", str(NBND))
    print()
    print("     Will run in", NPR, " processors")
    print("     Working directory:", WORKDIR)
    print("     Name of directory for wfc:", WFCDIRECTORY)
    try:
        BERRYPATH = str(os.environ["BERRYPATH"])
    except KeyError:
        BERRYPATH = str(os.path.dirname(os.path.dirname(__file__)))
    if BERRYPATH[-1] != "/":
        BERRYPATH = BERRYPATH + "/"
    print("     Path of BERRY files", BERRYPATH)
    print()
    print("     DFT calculations will be done on", PROGRAM)
    print("     The DFT files will be on:", DFTDIRECTORY)
    print("     DFT pseudopotential directory:", PSEUDODIR)
    print("     Name of scf file:", NAMESCF)
    print("     Name of nscf file:", NAMENSCF)
    print("     DFT prefix:", PREFIX)
    print("     DFT outdir:", OUTDIR)
    DFTDATAFILE = OUTDIR + PREFIX + ".xml"
    print("     DFT data file:", DFTDATAFILE)
    print()
    print("     Finished reading input file")
    print()

    sys.stdout.flush()

    if NPR == 1:
        MPI = ""
    else:
        MPI = "mpirun -np " + str(NPR) + " "

    # Calculates the k-points of the nscf mesh using data from the input file
    NSCFKPOINTS = ""  # to append to nscf file
    NKS = nkx * nky * nkz  # total number of k-points
    KPOINTS = np.zeros((NKS, 3), dtype=float)
    NKTOIJL = np.zeros((NKS, 3), dtype=int)  # Converts index nk to indices i,j,l
    IJLTONK = np.zeros((nkx, nky, nkz), dtype=int)  # Converts indices i,j,l to index nk

    nk = 0  # count the k-points
    for l in range(nkz):
        for j in range(nky):
            for i in range(nkx):
                k1, k2, k3 = (
                    round(K0[0] + step * i, 8),
                    round(K0[1] + step * j, 8),
                    round(K0[2] + step * l, 8),
                )
                kkk = (
                    "{:.7f}".format(k1)
                    + "  "
                    + "{:.7f}".format(k2)
                    + "  "
                    + "{:.7f}".format(k3)
                )
                NSCFKPOINTS = (
                    NSCFKPOINTS + kkk + "  1\n"
                )  # Build KPOINTS for nscf calculation
                KPOINTS[nk, 0] = k1
                KPOINTS[nk, 1] = k2
                KPOINTS[nk, 2] = k3
                NKTOIJL[nk, 0] = i
                NKTOIJL[nk, 1] = j
                NKTOIJL[nk, 2] = l
                IJLTONK[i, j, l] = nk
                nk = nk + 1

    # Starts DFT calculations ##############################
    # Runs scf calculation  ** DFT
    dft.scf(MPI, DFTDIRECTORY, NAMESCF, OUTDIR, PSEUDODIR)

    # Creates template for nscf calculation  ** DFT
    NSCF = dft.template(DFTDIRECTORY, NAMESCF)

    # Runs nscf calculations for all k-points  ** DFT **
    dft.nscf(MPI, DFTDIRECTORY, NAMENSCF, NSCF, NKS, NSCFKPOINTS, NBND)
    sys.stdout.flush()

    print("     Extracting data from DFT calculations")
    print()

    # Reading DFT data from file ###########################
    TREE = ET.parse(DFTDATAFILE)
    ROOT = TREE.getroot()
    OUTPUT = ROOT.find("output")
    GENERAL = ROOT.find("general_info")
    #for child in ROOT[3]:
    #    print(child.tag, child.attrib)
    #    for child1 in child:
    #        print(' ',child1.tag,child1.attrib)
    #        for child2 in child1:
    #            print('   ',child2.tag,child2.attrib)
    #    print()
    DFTVERSION = str(GENERAL.find("creator").attrib["VERSION"])
    DFTPROGRAM = str(GENERAL.find("creator").attrib["NAME"])
    print("     DFT version", DFTPROGRAM, DFTVERSION)
    EXTRACTVERSION = float(DFTVERSION[0:3])
    if EXTRACTVERSION >= 6.7:
        WFCK2R = "wfck2r.oct"
    else:
        WFCK2R = "wfck2r.mat"

    print()
    print("     Lattice vectors in units of a0 (bohr)")
    A1, A2, A3 = [
        np.array(list(map(float, it.text.split())))
        for it in OUTPUT.find("atomic_structure").find("cell")
    ]
    print("        a1:", A1)
    print("        a2:", A2)
    print("        a3:", A3)
    print()

    print("     Reciprocal lattice vectors in units of 2pi/a0 (2pi/bohr)")
    B1, B2, B3 = [
        np.array(list(map(float, it.text.split())))
        for it in OUTPUT.find("basis_set").find("reciprocal_lattice")
    ]
    print("        b1:", B1)
    print("        b2:", B2)
    print("        b3:", B3)
    print()

    print("     Number of points in real space in each direction")
    NR1 = int(OUTPUT.find("basis_set").find("fft_grid").attrib["nr1"])
    NR2 = int(OUTPUT.find("basis_set").find("fft_grid").attrib["nr2"])
    NR3 = int(OUTPUT.find("basis_set").find("fft_grid").attrib["nr3"])
    NR = NR1 * NR2 * NR3
    print("        nr1:", NR1)
    print("        nr2:", NR2)
    print("        nr3:", NR3)
    print("         nr:", NR)
    REFRPOINT = int(POINT * NR1 * NR2)
    print("     Point where phases match: ", str(REFRPOINT))
    print()

    NBND = int(OUTPUT.find("band_structure").find("nbnd").text)
    print("     Number of bands in the DFT calculation: ", NBND)
    NELEC = float(OUTPUT.find("band_structure").find("nelec").text)
    print("     Number of electrons: ", NELEC)
    NKS = int(OUTPUT.find("band_structure").find("nks").text)
    print("     Number of k-points in the DFT calculation: ", NKS)
    if str(OUTPUT.find("band_structure").find("noncolin").text) == "false":
        NONCOLINEAR = False
    else:
        NONCOLINEAR = True
    print("     Noncolinear calculation: ", NONCOLINEAR)
    if str(OUTPUT.find("band_structure").find("lsda").text) == "false":
        LSDA = False
    else:
        LSDA = True
    print("     Spin polarized calculation: ", LSDA)

    print()

    # for child in ROOT[3][9]:
    #  print(child.tag, child.attrib,child.text)
    #  for child1 in child:
    #    print(' ',child1.tag,child1.attrib,child1.text)

    EIGENVALUES = 2 * np.array(
        [
            list(map(float, it.text.split()))
            for it in OUTPUT.find("band_structure").iter("eigenvalues")
        ]
    )
    # print(EIGENVALUES)

    OCCUPATIONS = np.array(
        [
            list(map(float, it.text.split()))
            for it in OUTPUT.find("band_structure").iter("occupations")
        ]
    )
    # print(OCCUPATIONS)
    # Finished reading data from DFT files

    # Final calculations #########################################
    print()

    COUNT = 0
    RPOINT = np.zeros((NR, 3), dtype=float)
    for l in range(NR3):
        for k in range(NR2):
            for i in range(NR1):
                RPOINT[COUNT] = A1 * i / NR1 + A2 * k / NR2 + A3 * l / NR3
                COUNT += 1

    PHASE = np.exp(1j * np.dot(RPOINT, np.transpose(KPOINTS)))

    # Start saving data to files ##################################
    with open("phase.npy", "wb") as fich:
        np.save(fich, PHASE)
    fich.close()
    print("     PHASE saved to file phase.npy")

    NEIG = np.full((NKS, 4), -1, dtype=int)
    nk = -1
    with open("neighbors.dat", "w") as nei:
        for j in range(nky):
            for i in range(nkx):
                nk = nk + 1
                if i == 0:
                    n0 = -1
                else:
                    n0 = nk - 1
                if j == 0:
                    n1 = -1
                else:
                    n1 = nk - nkx
                if i == nkx - 1:
                    n2 = -1
                else:
                    n2 = nk + 1
                if j == nky - 1:
                    n3 = -1
                else:
                    n3 = nk + nkx
                nei.write(
                    str(nk)
                    + "  "
                    + str(n0)
                    + "  "
                    + str(n1)
                    + "  "
                    + str(n2)
                    + "  "
                    + str(n3)
                    + "\n"
                )
                NEIG[nk, 0] = n0
                NEIG[nk, 1] = n1
                NEIG[nk, 2] = n2
                NEIG[nk, 3] = n3
    nei.close()
    print("     Neighbors saved to file neighbors.dat")
    with open("neighbors.npy", "wb") as nnn:
        np.save(nnn, NEIG)
    nnn.close()
    print("     Neighbors saved to file neighbors.npy")

    # Save EIGENVALUES to file (in Ha)
    with open("eigenvalues.npy", "wb") as fich:
        np.save(fich, EIGENVALUES)
    fich.close()
    print("     EIGENVALUES saved to file eigenvalues.npy (Ry)")

    # Save OCCUPATIONS to file
    with open("occupations.npy", "wb") as fich:
        np.save(fich, OCCUPATIONS)
    fich.close()
    print("     OCCUPATIONS saved to file occupations.npy")

    # Save positions to file
    with open("positions.npy", "wb") as fich:
        np.save(fich, RPOINT)
    fich.close()
    print("     Positions saved to file positions.npy (bohr)")

    # Save kpoints to file
    with open("kpoints.npy", "wb") as fich:
        np.save(fich, KPOINTS)
    fich.close()
    print("     KPOINTS saved to file kpoints.npy (2pi/bohr)")

    # Save nktoijl to file
    with open("nktoijl.npy", "wb") as fich:
        np.save(fich, NKTOIJL)
    fich.close()
    print("     NKTOIJL saved to file nktoijl.npy, with convertion from nk to ijl")

    # Save ijltonk to file
    with open("ijltonk.npy", "wb") as fich:
        np.save(fich, IJLTONK)
    fich.close()
    print("     IJLTONK saved to file ijltonk.npy, with convertion from ijl to nk")

    # Save data to file 'datafile.npy'
    with open("datafile.npy", "wb") as fich:
        np.save(fich, K0)  # Initial k-point
        np.save(fich, nkx)  # Number of k-points in the x direction
        np.save(fich, nky)  # Number of k-points in the y direction
        np.save(fich, nkz)  # Number of k-points in the z direction
        np.save(fich, NKS)  # Total number of k-points
        np.save(fich, step)  # Step between k-points
        np.save(fich, NPR)  # Number of processors for the run
        np.save(fich, DFTDIRECTORY)  # Directory of DFT files
        np.save(fich, NAMESCF)  # Name of scf file (without suffix)
        np.save(fich, NAMENSCF)  # Name of nscf file (without suffix)
        np.save(fich, WFCDIRECTORY)  # Directory for the wfc files
        np.save(fich, PREFIX)  # Prefix of the DFT QE calculations
        np.save(fich, OUTDIR)  # Directory for DFT saved files
        np.save(fich, DFTDATAFILE)  # Path to DFT file with data of the run
        np.save(fich, A1)  # First lattice vector in real space
        np.save(fich, A2)  # Second lattice vector in real space
        np.save(fich, A3)  # Third lattice vector in real space
        np.save(fich, B1)  # First lattice vector in reciprocal space
        np.save(fich, B2)  # Second lattice vector in reciprocal space
        np.save(fich, B3)  # Third lattice vector in reciprocal space
        np.save(fich, NR1)  # Number of points of wfc in real space x direction
        np.save(fich, NR2)  # Number of points of wfc in real space y direction
        np.save(fich, NR3)  # Number of points of wfc in real space z direction
        np.save(fich, NR)  # Total number of points of wfc in real space
        np.save(fich, NBND)  # Number of bands
        np.save(fich, BERRYPATH)  # Path of BERRY files
        np.save(fich, REFRPOINT)  # Point in real space where all phases match
        np.save(fich, WORKDIR)  # Working directory
        np.save(fich, NONCOLINEAR)  # If the calculation is noncolinear
        np.save(fich, PROGRAM)  # DFT software to be used
        np.save(fich, LSDA)  # Spin polarized calculation
        np.save(fich, NELEC)  # Number of electrons
        np.save(fich, PREFIX)  # prefix of the DFT calculations
        np.save(fich, WFCK2R)  # File for extracting DFT wfc to real space
        np.save(fich, __version__)  # Version of berry where data was created
        np.save(fich, refname)  # Unique reference for the run
        np.save(fich, "dummy")  # Saving space for future values and compatibility
        np.save(fich, "dummy")  # Saving space for future values and compatibility
        np.save(fich, "dummy")  # Saving space for future values and compatibility
        np.save(fich, "dummy")  # Saving space for future values and compatibility
        np.save(fich, "dummy")  # Saving space for future values and compatibility
    fich.close()
    print("     Data saved to file datafile.npy")

    list_kpoints(nkx, nky)

    ###################################################################################
    # Finished
    footer(contatempo.tempo(STARTTIME, time.time()))
