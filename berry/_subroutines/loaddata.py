""" This program reads data from the files where it was saved

"""

import numpy as np


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
    refname = str(np.load(fich))
    vb = int(np.load(fich))


# Read eigenvalues from file   eigenvalues = np.array(nks,nbnd)
try:
    eigenvalues = np.load("eigenvalues.npy")
except IOError:
    print("  WARNING: No eigenvalues.npy file.")

# Read occupations from file   occupations = np.array(nks,nbnd)
try:
    occupations = np.load("occupations.npy")
except IOError:
    print("  WARNING: No occupations.npy file.")


# Read phase from file         phase = np.zeros((nr,nks),dtype=complex)
try:
    phase = np.load("phase.npy")
except IOError:
    print("  WARNING: No phase.npy file.")

# Read neighbors from file     neig = np.full((nks,4),-1,dtype=int)
try:
    neighbors = np.load("neighbors.npy")
except IOError:
    print("  WARNING: No neighbors.npy file.")


# Read kpoints from file       kpoints = np.zeros((nks,3), dtype=float)
try:
    kpoints = np.load("kpoints.npy")
except IOError:
    print("  WARNING: No kpoints.npy file.")

# Read nktoijl from file       nktoijl = np.zeros((nks,3),dtype=int)
try:
    nktoijl = np.load("nktoijl.npy")
except IOError:
    print("  WARNING: No nktoijl.npy file.")

# Read ijltonk from file       ijltonk = np.zeros((nkx,nky,nkz),dtype=int)
try:
    ijltonk = np.load("ijltonk.npy")
except IOError:
    print("  WARNING: No ijltonk.npy file.")


# Read positions from file     r = np.zeros((nr,3), dtype=float)
try:
    r = np.load("positions.npy")
except IOError:
    print("  WARNING: No positions.npy file.")


# To call these variables from another program:
# import loaddata as d
# d.k0
# d.nks
# etc.
