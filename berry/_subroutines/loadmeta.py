""" This program reads all the data from the datafile.npy file

"""

import numpy as np

with open("data/datafile.npy", "rb") as fich:
    version = str(np.load(fich))  # Version of berry where data was created
    refname = str(np.load(fich))  # Unique reference for the run
    dimensions = int(np.load(fich)) # Number of dimensions

    workdir = str(np.load(fich))  # Working directory
    data_dir = str(np.load(fich)) # Directory for saving data
    log_dir = str(np.load(fich))  # Directory for the logs
    geometry_dir = str(np.load(fich))  # Directory for the Berry geometries

    k0 = np.load(fich) # Initial k-point
    nkx = int(np.load(fich))    # Number of k-points in the x direction
    nky = int(np.load(fich))    # Number of k-points in the y direction
    nkz = int(np.load(fich))    # Number of k-points in the z direction
    nks = int(np.load(fich))    # Total number of k-points
    step = float(np.load(fich))  # Step between k-points
    npr = int(np.load(fich))     # Number of processors for the run
    rpoint = int(np.load(fich))  # Point in real space where all phases match

    dftdirectory = str(np.load(fich)) # Directory of DFT files
    name_scf = str(np.load(fich)) # Name of scf file (without suffix)
    name_nscf = str(np.load(fich)) # Name of nscf file (without suffix)
    prefix = str(np.load(fich)) # Prefix of the DFT QE calculations
    wfcdirectory = str(np.load(fich))  # Directory for the wfc files
    outdir = str(np.load(fich))  # Directory for DFT saved files
    dftdatafile = str(np.load(fich)) # Path to DFT file with data of the run
    program = str(np.load(fich))  # DFT software to be used

    a1 = np.load(fich)  # First lattice vector in real space
    a2 = np.load(fich)  # Second lattice vector in real space
    a3 = np.load(fich)  # Third lattice vector in real space
    b1 = np.load(fich)  # First lattice vector in reciprocal space
    b2 = np.load(fich)  # Second lattice vector in reciprocal space
    b3 = np.load(fich)  # Third lattice vector in reciprocal space
    nr1 = int(np.load(fich))   # Number of points of wfc in real space x direction
    nr2 = int(np.load(fich))   # Number of points of wfc in real space y direction
    nr3 = int(np.load(fich))   # Number of points of wfc in real space z direction
    nr = int(np.load(fich))    # Total number of points of wfc in real space
    nbnd = int(np.load(fich))  # Number of bands
  
    noncolin = bool(np.load(fich)) # If the calculation is noncolinear
    lsda = bool(np.load(fich))     # Spin polarized calculation
    nelec = float(np.load(fich))   # Number of electrons
    wfck2r = str(np.load(fich))    # File for extracting DFT wfc to real space
    vb = int(np.load(fich))        # Valence band number

    kvector1 = np.load(fich)        # First vector that define volume in k space
    kvector2 = np.load(fich)        # Second vector that define volume in k space
    kvector3 = np.load(fich)        # Third vector that define volume in k space

    wfcut = int(np.load(fich))           # Cutoff band