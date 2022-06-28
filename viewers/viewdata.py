"""This program reads and prints the preprocessing data
"""

import loaddata as d

if __name__ == "__main__":

    print("    Initial k-point                                    ", d.k0)
    print("    Number of k-points in the x direction              ", d.nkx)
    print("    Number of k-points in the y direction              ", d.nky)
    print("    Number of k-points in the z direction              ", d.nkz)
    print("    Total number of k-points                           ", d.nks)
    print("    Step between k-points                              ", d.step)
    print("    Number of processors for the run                   ", d.npr)
    print("    Directory of DFT files                             ", d.dftdirectory)
    print("    Name of scf file (without suffix)                  ", d.name_scf)
    print("    Name of nscf file (without suffix)                 ", d.name_nscf)
    print("    Directory for the wfc files                        ", d.wfcdirectory)
    print("    Prefix of the DFT QE calculations                  ", d.prefix)
    print("    Directory for DFT saved files                      ", d.outdir)
    print("    Path to DFT file with data of the run              ", d.dftdatafile)
    print("    First lattice vector in real space                 ", d.a1)
    print("    Second lattice vector in real space                ", d.a2)
    print("    Third lattice vector in real space                 ", d.a3)
    print("    First lattice vector in reciprocal space           ", d.b1)
    print("    Second lattice vector in reciprocal space          ", d.b2)
    print("    Third lattice vector in reciprocal space           ", d.b3)
    print("    Number of points of wfc in real space x direction  ", d.nr1)
    print("    Number of points of wfc in real space y direction  ", d.nr2)
    print("    Number of points of wfc in real space z direction  ", d.nr3)
    print("    Total number of points of wfc in real space        ", d.nr)
    print("    Number of bands                                    ", d.nbnd)
    print("    Path of BERRY files                                ", d.berrypath)
    print("    Point in real space where all phases match         ", d.rpoint)
    print("    Working directory                                  ", d.workdir)
    print("    If the calculation is noncolinear                  ", d.noncolin)
    print("    DFT software to be used                            ", d.program)
    print("    Spin polarized calculation                         ", d.lsda)
    print("    Number of electrons                                ", d.nelec)
    print("    prefix of the DFT calculations                     ", d.prefix)
    print("    File for extracting DFT wfc to real space          ", d.wfck2r)
    print("    Version of berry where data was created            ", d.version)