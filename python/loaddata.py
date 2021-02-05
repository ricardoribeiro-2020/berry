###################################################################################
# This program reads data from the files where it was saved
###################################################################################

# This is to include maths
import numpy as np

# To read the data from file:
with open('datafile.npy', 'rb') as f:
  k0           = np.load(f)
  nkx          = np.load(f)
  nky          = np.load(f)
  nkz          = np.load(f)
  nks          = np.load(f)
  step         = np.load(f)
  npr          = np.load(f)
  dftdirectory = np.load(f)
  name_scf     = np.load(f)
  name_nscf    = np.load(f)
  wfcdirectory = np.load(f)
  prefix       = np.load(f)
  outdir       = np.load(f)
  dftdatafile  = np.load(f)
  a1           = np.load(f)
  a2           = np.load(f)
  a3           = np.load(f)
  b1           = np.load(f)
  b2           = np.load(f)
  b3           = np.load(f)
  nr1          = np.load(f)
  nr2          = np.load(f)
  nr3          = np.load(f)
  nr           = np.load(f)
  nbnd         = np.load(f)
  berrypath    = np.load(f)
  rpoint       = np.load(f)
  workdir      = np.load(f)
  noncolin     = np.load(f)
  program      = np.load(f)
  lsda         = np.load(f)
  nelec        = np.load(f)
f.close()


# Read eigenvalues from file   eigenvalues = np.array(nks,nbnd)
with open('eigenvalues.npy', 'rb') as f: 
  eigenvalues = np.load(f)
f.close()


# Read occupations from file   occupations = np.array(nks,nbnd)
with open('occupations.npy', 'rb') as f:
  occupations = np.load(f)
f.close()


# Read phase from file         phase = np.zeros((nr,nks),dtype=complex)
with open('phase.npy','rb') as f:
  phase = np.load(f)
f.close()


# Read neighbors from file     neig = np.full((nks,4),-1,dtype=int)
with open('neighbors.npy','rb') as f:
  neighbors = np.load(f)
f.close()


# Read kpoints from file       kpoints = np.zeros((nks,3), dtype=float)
with open('kpoints.npy', 'rb') as f:
  kpoints = np.load(f)
f.close()

# Read nktoijl from file       nktoijl = np.zeros((nks,3),dtype=int)
with open('nktoijl.npy', 'rb') as f:
  nktoijl = np.load(f)
f.close()

# Read ijltonk from file       ijltonk = np.zeros((nkx,nky,nkz),dtype=int)
with open('ijltonk.npy', 'rb') as f:
  ijltonk = np.load(f)
f.close()


# Read positions from file     r = np.zeros((nr,3), dtype=float)
with open('positions.npy', 'rb') as f:
  r = np.load(f)
f.close()


# To call these variables from another program:
# import loaddata as d
# d.k0
# d.nks
# etc.


