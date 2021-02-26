###################################################################################
# This program reads data from the files where it was saved
###################################################################################

import numpy as np

# To read the data from file:
with open('datafile.npy', 'rb') as fich:
    k0           = np.load(fich)
    nkx          = np.load(fich)
    nky          = np.load(fich)
    nkz          = np.load(fich)
    nks          = np.load(fich)
    step         = np.load(fich)
    npr          = np.load(fich)
    dftdirectory = np.load(fich)
    name_scf     = np.load(fich)
    name_nscf    = np.load(fich)
    wfcdirectory = np.load(fich)
    prefix       = np.load(fich)
    outdir       = np.load(fich)
    dftdatafile  = np.load(fich)
    a1           = np.load(fich)
    a2           = np.load(fich)
    a3           = np.load(fich)
    b1           = np.load(fich)
    b2           = np.load(fich)
    b3           = np.load(fich)
    nr1          = np.load(fich)
    nr2          = np.load(fich)
    nr3          = np.load(fich)
    nr           = np.load(fich)
    nbnd         = np.load(fich)
    berrypath    = np.load(fich)
    rpoint       = np.load(fich)
    workdir      = np.load(fich)
    noncolin     = np.load(fich)
    program      = np.load(fich)
    lsda         = np.load(fich)
    nelec        = np.load(fich)
    prefix       = np.load(fich)
    outdir       = np.load(fich)
fich.close()


# Read eigenvalues from file   eigenvalues = np.array(nks,nbnd)
with open('eigenvalues.npy', 'rb') as fich:
    eigenvalues = np.load(fich)
fich.close()


# Read occupations from file   occupations = np.array(nks,nbnd)
with open('occupations.npy', 'rb') as fich:
    occupations = np.load(fich)
fich.close()


# Read phase from file         phase = np.zeros((nr,nks),dtype=complex)
with open('phase.npy', 'rb') as fich:
    phase = np.load(fich)
fich.close()


# Read neighbors from file     neig = np.full((nks,4),-1,dtype=int)
with open('neighbors.npy', 'rb') as fich:
    neighbors = np.load(fich)
fich.close()


# Read kpoints from file       kpoints = np.zeros((nks,3), dtype=float)
with open('kpoints.npy', 'rb') as fich:
    kpoints = np.load(fich)
fich.close()

# Read nktoijl from file       nktoijl = np.zeros((nks,3),dtype=int)
with open('nktoijl.npy', 'rb') as fich:
    nktoijl = np.load(fich)
fich.close()

# Read ijltonk from file       ijltonk = np.zeros((nkx,nky,nkz),dtype=int)
with open('ijltonk.npy', 'rb') as fich:
    ijltonk = np.load(fich)
fich.close()


# Read positions from file     r = np.zeros((nr,3), dtype=float)
with open('positions.npy', 'rb') as fich:
    r = np.load(fich)
fich.close()


# To call these variables from another program:
# import loaddata as d
# d.k0
# d.nks
# etc.
