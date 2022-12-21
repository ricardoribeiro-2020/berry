from colorama import Fore, Back, Style

import numpy as np

from berry._subroutines.write_k_points import _float_numbers
import berry._subroutines.loadmeta as m
import berry._subroutines.loaddata as d


Options = str

def log_data():
    print("#"*100)
    print("# DATA OF THE SYSTEM")
    print("#"*100)
    print(f"\n\tInitial k-point                                  {m.k0}")
    print(f"\tNumber of k-points in the x direction              {m.nkx}")
    print(f"\tNumber of k-points in the y direction              {m.nky}")
    print(f"\tNumber of k-points in the z direction              {m.nkz}")
    print(f"\tTotal number of k-points                           {m.nks}")
    print(f"\tStep between k-points                              {m.step}")
    print(f"\tNumber of processors for the run                   {m.npr}")
    print(f"\tDirectory of DFT files                             {m.dftdirectory}")
    print(f"\tName of scf file (without suffix)                  {m.name_scf}")
    print(f"\tName of nscf file (without suffix)                 {m.name_nscf}")
    print(f"\tDirectory for the wfc files                        {m.wfcdirectory}")
    print(f"\tPrefix of the DFT QE calculations                  {m.prefix}")
    print(f"\tDirectory for DFT saved files                      {m.outdir}")
    print(f"\tPath to DFT file with data of the run              {m.dftdatafile}")
    print(f"\tFirst lattice vector in real space                 {m.a1}")
    print(f"\tSecond lattice vector in real space                {m.a2}")
    print(f"\tThird lattice vector in real space                 {m.a3}")
    print(f"\tFirst lattice vector in reciprocal space           {m.b1}")
    print(f"\tSecond lattice vector in reciprocal space          {m.b2}")
    print(f"\tThird lattice vector in reciprocal space           {m.b3}")
    print(f"\tNumber of points of wfc in real space x direction  {m.nr1}")
    print(f"\tNumber of points of wfc in real space y direction  {m.nr2}")
    print(f"\tNumber of points of wfc in real space z direction  {m.nr3}")
    print(f"\tTotal number of points of wfc in real space        {m.nr}")
    print(f"\tNumber of bands                                    {m.nbnd}")
    print(f"\tPath of BERRY files                                {m.berrypath}")
    print(f"\tPoint in real space where all phases match         {m.rpoint}")
    print(f"\tWorking directory                                  {m.workdir}")
    print(f"\tIf the calculation is noncolinear                  {m.noncolin}")
    print(f"\tDFT software to be used                            {m.program}")
    print(f"\tSpin polarized calculation                         {m.lsda}")
    print(f"\tNumber of electrons                                {m.nelec}")
    print(f"\tprefix of the DFT calculations                     {m.prefix}")
    print(f"\tFile for extracting DFT wfc to real space          {m.wfck2r}")
    print(f"\tVersion of berry where data was created            {m.version}\n")


def log_dot1():
    print("#"*100)
    print("# DOT PRODUCT OF THE WAVEFUNCTIONS (option 1)")
    print("#"*100)
    print(f"\n\tDirectory where the wfc are: {m.wfcdirectory}")
    print(f"\tNumber of k-points in each direction: {m.nkx}, {m.nky}, {m.nkz}")
    print(f"\tTotal number of k-points: {m.nks}")
    print(f"\tNumber of bands: {m.nbnd}\n")

    try:
        connections = np.load("dp.npy")
    except FileNotFoundError:
        raise FileNotFoundError("File dp.npy not found. Run the script dotproduct.py to create it.")

    print(f"\n\tShape of connections array: {connections.shape}")

    for nk in range(m.nks):
        for j in range(4):
            neighbor = d.neighbors[nk, j]
            print(f"\t{nk=}; {neighbor=}")
            for band in range(m.nbnd):
                line = "  "
                for band1 in range(m.nbnd):
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

    print("     Directory where the wfc are:", m.wfcdirectory)
    print("     Number of k-points in each direction:", m.nkx, m.nky, m.nkz)
    print("     Total number of k-points:", m.nks)
    print("     Number of bands:", m.nbnd)
    print("     Band calculated:", band1)

    connections = np.load("dp.npy")
    valuesarray = np.zeros((m.nks, m.nbnd), dtype=float)
    bandsarray = np.zeros((m.nks, m.nbnd), dtype=int)
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
        for i in range(m.nks):
            nk = d.neighbors[i, j]
            for band in range(m.nbnd):
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
        for _ in range(m.nky):
            lin = ""
            print()
            for _ in range(m.nkx):
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
    print(_float_numbers(m.nkx, m.nky, d.eigenvalues[:, band], acc))


def log_neighbors():
    print("#"*100)
    print("# NEIGHBORS")
    print("#"*100)
    m_ = np.max(d.neighbors)
    for nk in range(m.nks):
        print(f"{nk:5}:\t{d.neighbors[nk, 0]:5} {d.neighbors[nk, 1]:5} {d.neighbors[nk, 1]:5} {d.neighbors[nk, 3]:5}")


def log_occupation():
    print("#"*100)
    print("# OCCUPATION")
    print("#"*100)
    for nk in range(m.nks):
        print(nk, d.occupations[nk, :])


def log_r_space():
    print("#"*100)
    print("# R-SPACE")
    print("#"*100)
    for i in range(m.nr):
        print(i, d.r[i, 0], d.r[i, 1], d.r[i, 2])


def debug(args):
    if args.debug_vis == "data":
        log_data()
    elif args.debug_vis == "dot1":
        log_dot1()
    elif args.debug_vis == "dot2":
        log_dot2(args.band)
    elif args.debug_vis == "eigen":
        log_eigen(args.band, args.acc)
    elif args.debug_vis == "occ":
        log_occupation()
    elif args.debug_vis == "neig":
        log_neighbors()
    elif args.debug_vis == "r-space":
        log_r_space()

if __name__ == "__main__":
    debug()
