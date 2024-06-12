""" This program reads data from the files where it was saved

"""
import numpy as np # type: ignore

# Read eigenvalues from file   eigenvalues = np.array(nks,nbnd)
try:
    eigenvalues = np.load("data/eigenvalues.npy")
except IOError:
    print("  WARNING: No eigenvalues.npy file.")

# Read occupations from file   occupations = np.array(nks,nbnd)
try:
    occupations = np.load("data/occupations.npy")
except IOError:
    print("  WARNING: No occupations.npy file.")

# Read neighbors from file     neig = np.full((nks,4),-1,dtype=int)
try:
    neighbors = np.load("data/neighbors.npy")
except IOError:
    print("  WARNING: No neighbors.npy file.")


# Read kpoints from file       kpoints = np.zeros((nks,3), dtype=float)
try:
    kpoints = np.load("data/kpoints.npy")
except IOError:
    print("  WARNING: No kpoints.npy file.")

# Read nktoijl from file       nktoijl = np.zeros((nks,3),dtype=int)
try:
    nktoijl = np.load("data/nktoijl.npy")
except IOError:
    print("  WARNING: No nktoijl.npy file.")

# Read ijltonk from file       ijltonk = np.zeros((nkx,nky,nkz),dtype=int)
try:
    ijltonk = np.load("data/ijltonk.npy")
except IOError:
    print("  WARNING: No ijltonk.npy file.")


# Read positions from file     r = np.zeros((nr,3), dtype=float)
try:
    r = np.load("data/positions.npy")
except IOError:
    print("  WARNING: No positions.npy file.")


# To call these variables from another program:
# import loaddata as d
# d.k0
# d.nks
# etc.
