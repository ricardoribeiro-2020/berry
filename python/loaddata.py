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
f.close()


with open('eigenvalues.npy', 'rb') as f:
  eigenvalues = np.load(f)
f.close()


# Read occupations from file
with open('occupations.npy', 'rb') as f:
  occupations = np.load(f)
f.close()


# Read phase from file
with open('phase.npy','rb') as f:
  phase = np.load(f)
f.close()


# Read neighbors from file
with open('neighbors.npy','rb') as f:
  neighbors = np.load(f)
f.close()


# To call these variables from another program:
# import loaddata as d
# d.k0
# d.nks
# etc.


