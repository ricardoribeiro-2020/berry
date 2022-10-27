from colorama import Fore, Back, Style

import numpy as np

from berry.cli import viz_debug_cli
from berry._subroutines.write_k_points import _float_numbers

import berry._subroutines.loaddata as d


Options = str

def log_data():
    print("#"*100)
    print("# DATA OF THE SYSTEM")
    print("#"*100)
    print(f"\n\tInitial k-point                                    {d.k0}")
    print(f"\tNumber of k-points in the x direction              {d.nkx}")
    print(f"\tNumber of k-points in the y direction              {d.nky}")
    print(f"\tNumber of k-points in the z direction              {d.nkz}")
    print(f"\tTotal number of k-points                           {d.nks}")
    print(f"\tStep between k-points                              {d.step}")
    print(f"\tNumber of processors for the run                   {d.npr}")
    print(f"\tDirectory of DFT files                             {d.dftdirectory}")
    print(f"\tName of scf file (without suffix)                  {d.name_scf}")
    print(f"\tName of nscf file (without suffix)                 {d.name_nscf}")
    print(f"\tDirectory for the wfc files                        {d.wfcdirectory}")
    print(f"\tPrefix of the DFT QE calculations                  {d.prefix}")
    print(f"\tDirectory for DFT saved files                      {d.outdir}")
    print(f"\tPath to DFT file with data of the run              {d.dftdatafile}")
    print(f"\tFirst lattice vector in real space                 {d.a1}")
    print(f"\tSecond lattice vector in real space                {d.a2}")
    print(f"\tThird lattice vector in real space                 {d.a3}")
    print(f"\tFirst lattice vector in reciprocal space           {d.b1}")
    print(f"\tSecond lattice vector in reciprocal space          {d.b2}")
    print(f"\tThird lattice vector in reciprocal space           {d.b3}")
    print(f"\tNumber of points of wfc in real space x direction  {d.nr1}")
    print(f"\tNumber of points of wfc in real space y direction  {d.nr2}")
    print(f"\tNumber of points of wfc in real space z direction  {d.nr3}")
    print(f"\tTotal number of points of wfc in real space        {d.nr}")
    print(f"\tNumber of bands                                    {d.nbnd}")
    print(f"\tPath of BERRY files                                {d.berrypath}")
    print(f"\tPoint in real space where all phases match         {d.rpoint}")
    print(f"\tWorking directory                                  {d.workdir}")
    print(f"\tIf the calculation is noncolinear                  {d.noncolin}")
    print(f"\tDFT software to be used                            {d.program}")
    print(f"\tSpin polarized calculation                         {d.lsda}")
    print(f"\tNumber of electrons                                {d.nelec}")
    print(f"\tprefix of the DFT calculations                     {d.prefix}")
    print(f"\tFile for extracting DFT wfc to real space          {d.wfck2r}")
    print(f"\tVersion of berry where data was created            {d.version}\n")


def log_dot1():
    print("#"*100)
    print("# DOT PRODUCT OF THE WAVEFUNCTIONS (option 1)")
    print("#"*100)
    print(f"\n\tDirectory where the wfc are: {d.wfcdirectory}")
    print(f"\tNumber of k-points in each direction: {d.nkx}, {d.nky}, {d.nkz}")
    print(f"\tTotal number of k-points: {d.nks}")
    print(f"\tNumber of bands: {d.nbnd}\n")

    try:
        connections = np.load("dp.npy")
    except FileNotFoundError:
        raise FileNotFoundError("File dp.npy not found. Run the script dotproduct.py to create it.")

    print(f"\n\tShape of connections array: {connections.shape}")

    for nk in range(d.nks):
        for j in range(4):
            neighbor = d.neighbors[nk, j]
            print(f"\t{nk=}; {neighbor=}")
            for band in range(d.nbnd):
                line = "  "
                for band1 in range(d.nbnd):
                    if connections[nk, j, band, band1] > 0.1:
                        line = (
                            line
                            + "{:0.1f}".format(connections[nk, j, band, band1])
                            + " "
                        )
                    else:
                        line = line + "    "
                print(line)


def log_dot2(band1: int):
    print("#"*100)
    print("# DOT PRODUCT OF THE WAVEFUNCTIONS (option 2)")
    print("#"*100)

    print("     Directory where the wfc are:", d.wfcdirectory)
    print("     Number of k-points in each direction:", d.nkx, d.nky, d.nkz)
    print("     Total number of k-points:", d.nks)
    print("     Number of bands:", d.nbnd)
    print("     Band calculated:", band1)

    connections = np.load("dp.npy")
    valuesarray = np.zeros((d.nks, d.nbnd), dtype=float)
    bandsarray = np.zeros((d.nks, d.nbnd), dtype=int)
    #    print(connections.shape)

    coresback = [
        Back.BLACK,
        Back.BLUE,
        Back.GREEN,
        Back.RED,
        Back.YELLOW,
        Back.MAGENTA,
        Back.CYAN,
        Back.WHITE,
        Back.BLACK,
        Back.RED,
        Back.GREEN,
        Back.YELLOW,
        Back.BLUE,
        Back.MAGENTA,
        Back.CYAN,
        Back.WHITE,
        Back.BLACK,
        Back.RED,
        Back.GREEN,
        Back.YELLOW,
        Back.BLUE,
        Back.MAGENTA,
        Back.CYAN,
        Back.WHITE,
    ]
    coresfore = [
        Fore.WHITE,
        Fore.WHITE,
        Fore.WHITE,
        Fore.BLACK,
        Fore.WHITE,
        Fore.WHITE,
        Fore.BLACK,
        Fore.BLACK,
        Fore.WHITE,
        Fore.WHITE,
        Fore.WHITE,
        Fore.WHITE,
        Fore.BLACK,
        Fore.WHITE,
        Fore.BLACK,
        Fore.BLACK,
        Fore.WHITE,
        Fore.WHITE,
        Fore.WHITE,
        Fore.BLACK,
        Fore.WHITE,
        Fore.WHITE,
        Fore.BLACK,
        Fore.BLACK,
    ]

    for j in range(4):
        for i in range(d.nks):
            nk = d.neighbors[i, j]
            for band in range(d.nbnd):
                bandsarray[i, band] = np.argmax(connections[i, j, band, :])
                valuesarray[i, band] = np.amax(connections[i, j, band, :])

        nk = -1
        SEP = " "
        precision = 1
        print(
            "         | y  x ->  "
            + coresback[0]
            + coresfore[0]
            + " 0 "
            + coresback[1]
            + coresfore[1]
            + " 1 "
            + coresback[2]
            + coresfore[2]
            + " 2 "
            + coresback[3]
            + coresfore[3]
            + " 3 "
            + coresback[4]
            + coresfore[4]
            + " 4 "
            + coresback[5]
            + coresfore[5]
            + " 5 "
            + coresback[6]
            + coresfore[6]
            + " 6 "
            + coresback[7]
            + coresfore[7]
            + " 7 "
            + Style.RESET_ALL
        )
        for _ in range(d.nky):
            lin = ""
            print()
            for _ in range(d.nkx):
                nk = nk + 1
                val = (
                    coresback[bandsarray[nk, band1]]
                    + coresfore[bandsarray[nk, band1]]
                    + "{0:.{1}f}".format(valuesarray[nk, band1], precision)
                    + Style.RESET_ALL
                )
                if valuesarray[nk, band1] < 0:
                    lin += SEP + str(val)
                elif 0 <= valuesarray[nk, band1] < 10:
                    lin += SEP + SEP + str(val)
                elif 9 < valuesarray[nk, band1] < 100:
                    lin += SEP + str(val)
            print(lin)

        print()
        print()


def log_eigen(band: int, acc: int):
    print("#"*100)
    print("# EIGENVALUES")
    print("#"*100)
    _float_numbers(d.nkx, d.nky, d.eigenvalues[:, band], acc)


def log_neighbors():
    print("#"*100)
    print("# NEIGHBORS")
    print("#"*100)
    m = np.max(d.neighbors)
    for nk in range(d.nks):
        print(f"{nk:5}:\t{d.neighbors[nk, 0]:5} {d.neighbors[nk, 1]:5} {d.neighbors[nk, 1]:5} {d.neighbors[nk, 3]:5}")


def log_occupation():
    print("#"*100)
    print("# OCCUPATION")
    print("#"*100)
    for nk in range(d.nks):
        print(nk, d.occupations[nk, :])


def log_r_space():
    print("#"*100)
    print("# R-SPACE")
    print("#"*100)
    for i in range(d.nr):
        print(i, d.r[i, 0], d.r[i, 1], d.r[i, 2])


def debug():
    args = viz_debug_cli()

    if args.viz_prog == "data":
        log_data()
    elif args.viz_prog == "dot1":
        log_dot1()
    elif args.viz_prog == "dot2":
        log_dot2(args.band)
    elif args.viz_prog == "eigen":
        log_eigen(args.band, args.acc)
    elif args.viz_prog == "occ":
        log_occupation()
    elif args.viz_prog == "neig":
        log_neighbors()
    elif args.viz_prog == "r-space":
        log_r_space()

if __name__ == "__main__":
    debug()
