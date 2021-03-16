""" This program reads data from the files where it was saved

"""

import numpy as np

# pylint: disable=C0103
###################################################################################

# To read the data from file:
with open("datafile.npy", "rb") as fich:
    k0 = np.load(fich)
    nkx = int(np.load(fich))
    nky = int(np.load(fich))
    nkz = int(np.load(fich))
    nks = int(np.load(fich))
    step = float(np.load(fich))
    npr = int(np.load(fich))
    dftdirectory = str(np.load(fich))
    name_scf = str(np.load(fich))
    name_nscf = str(np.load(fich))
    wfcdirectory = str(np.load(fich))
    prefix = str(np.load(fich))
    outdir = str(np.load(fich))
    dftdatafile = str(np.load(fich))
    a1 = np.load(fich)
    a2 = np.load(fich)
    a3 = np.load(fich)
    b1 = np.load(fich)
    b2 = np.load(fich)
    b3 = np.load(fich)
    nr1 = int(np.load(fich))
    nr2 = int(np.load(fich))
    nr3 = int(np.load(fich))
    nr = int(np.load(fich))
    nbnd = int(np.load(fich))
    berrypath = str(np.load(fich))
    rpoint = int(np.load(fich))
    workdir = str(np.load(fich))
    noncolin = str(np.load(fich))
    program = str(np.load(fich))
    lsda = str(np.load(fich))
    nelec = float(np.load(fich))
    prefix = str(np.load(fich))
    wfck2r = str(np.load(fich))
    version = str(np.load(fich))
fich.close()


# Read eigenvalues from file   eigenvalues = np.array(nks,nbnd)
with open("eigenvalues.npy", "rb") as fich:
    eigenvalues = np.load(fich)
fich.close()


# Read occupations from file   occupations = np.array(nks,nbnd)
with open("occupations.npy", "rb") as fich:
    occupations = np.load(fich)
fich.close()


# Read phase from file         phase = np.zeros((nr,nks),dtype=complex)
with open("phase.npy", "rb") as fich:
    phase = np.load(fich)
fich.close()


# Read neighbors from file     neig = np.full((nks,4),-1,dtype=int)
with open("neighbors.npy", "rb") as fich:
    neighbors = np.load(fich)
fich.close()


# Read kpoints from file       kpoints = np.zeros((nks,3), dtype=float)
with open("kpoints.npy", "rb") as fich:
    kpoints = np.load(fich)
fich.close()

# Read nktoijl from file       nktoijl = np.zeros((nks,3),dtype=int)
with open("nktoijl.npy", "rb") as fich:
    nktoijl = np.load(fich)
fich.close()

# Read ijltonk from file       ijltonk = np.zeros((nkx,nky,nkz),dtype=int)
with open("ijltonk.npy", "rb") as fich:
    ijltonk = np.load(fich)
fich.close()


# Read positions from file     r = np.zeros((nr,3), dtype=float)
with open("positions.npy", "rb") as fich:
    r = np.load(fich)
fich.close()


# To call these variables from another program:
# import loaddata as d
# d.k0
# d.nks
# etc.
