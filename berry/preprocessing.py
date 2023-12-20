from typing import Dict, Union, List, Tuple
from multiprocessing import Pool
from itertools import product

import os
import re
import time
import logging
import platform
import xml.etree.ElementTree as ET

import numpy as np

from berry import log, __version__
from berry._subroutines.parserQE import parser

# Time when the run starts, then used for reference to the run:
now = time.strftime("%d-%m-%Y_%H:%M:%S", time.gmtime())
Data = Dict[str, Union[int, str, float]]

# This class initializes a berry run
class Preprocess:
    def __init__(self, 
                 k0: List[float], 
                 nkx: int, 
                 nky: int, 
                 nkz: int, 
                 step: float, 
                 nbnd: int, 
                 logger_name: str = "preprocess", 
                 logger_level: int = logging.INFO, 
                 npr: int = 1, 
                 dft_dir: str = "dft", 
                 scf: str = "scf.in", 
                 nscf: str = "", 
                 wfc_dir: str = "wfc", 
                 point: float = 1.178097, 
                 program: str = "QE", 
                 ref_name: str = now, 
                 flush: bool = False
                ):
        
        self.work_dir = os.getcwd() + "/"    # Define working directory
        self.k0 = k0                         # Coordinates of first k-point in reciprocal space
        self.nkx = nkx                       # Number of k-points in the x direction
        self.nky = nky                       # Number of k-points in the y direction
        self.nkz = nkz                       # Number of k-points in the z direction
        self.step = step                     # Distance between k-points
        self.nbnd = nbnd                     # Number of bands to be included in the calculation
        self.npr = npr                       # Number of processes to be run
        self.dft_dir = dft_dir               # Directory where the DFT files go
        self.scf = scf                       # Name of the scf DFT file to run
        self.nscf = nscf                     # Name of the nscf DFT file to run
        self.wfc_dir = wfc_dir               # Directory to save the wavefunctions
        self.point = point                   # Reference to point where wavefunctions will be set to the same phase
        self.program = program               # Name of the DFT software to be used
        self.ref_name = ref_name             # Unique reference for the whole run

        if self.nkz > 1 and self.nky > 1 and self.nkz > 1:
            self.dimensions = 3                  # Number of spatial dimensions of the material
        elif self.nkz == 1 and self.nky > 1:
            self.dimensions = 2
        else:
            self.dimensions = 1
        # If it is 2D, use x and y directions
        # If it is 1D, use x direction
        # Only these possibilities are available

        if not os.path.exists("log"):
            os.mkdir("log")                      # Creates log directory
        self.logger = log(logger_name, "PREPROCESS", level=logger_level, flush=flush)

        # Full path to the log directory:
        self.log_dir = os.path.join(self.work_dir, "log")
        # Full path to the data directory:
        self.data_dir = os.path.join(self.work_dir, "data")

        self.nscf_kpoints = ""
        self.__mpi = "" if self.npr == 1 else f"mpirun -np {self.npr}"
        self.__nks = self.nkx * self.nky * self.nkz
        self.kpoints, self.nktijl, self.ijltonk = self._build_kpoints()

        # Full path to the dft directory
        self.dft_dir = os.path.join(self.work_dir, self.dft_dir)
        # Full path to the wfc directory
        self.wfc_dir = os.path.join(self.data_dir, self.wfc_dir)
        # Full path to the geometry directory
        self.geometry_dir = os.path.join(self.data_dir, "geometry")

        # Correct scf's file names if necessary
    #    if not self.scf.endswith(".in"):
    #        self.scf += ".in"
        # Create variable for the name of the nscf file
        self.nscf = "n" + os.path.basename(self.scf)

        # Create variable with full path for the DFT's files
        self.scf = os.path.join(self.dft_dir, self.scf)
        self.nscf = os.path.join(self.dft_dir, self.nscf)

        if self.program == "QE":    # If the DFT program is QE (only option for now)
            # Get outdir from scf
            try:
                self.out_dir = os.path.abspath(parser("outdir", self.scf))
            except IndexError:
                raise ValueError(f"outdir keyword not found in {self.scf}. Make sure your scf file has the 'outdir' keyword set to './'")
            # Get pseudo_dir from scf
            try:
                self.pseudo_dir = os.path.abspath(parser("pseudo_dir", self.scf))
            except IndexError:
                raise ValueError(f"pseudo_dir keyword not found in {self.scf}. Make sure your scf file has the 'pseudo_dir' keyword set.")
            # Get prefix from scf
            try:
                self.prefix = parser("prefix", self.scf)
            except IndexError:
                raise ValueError(f"prefix keyword not found in {self.scf}. Make sure your scf file has the 'prefix' keyword set.")
        
        # Full path to xml DFT file, with the data of the run
        self.dft_data_file = os.path.join(self.out_dir, self.prefix + ".xml")

        # Write header to the log file
        self.logger.header()
        self._log_inputs()
        self.logger.info("\tRunning a",self.dimensions,"dimensions material.\n")

    # Run a sequence of processes based on the data
    def run(self):
        # Create directories where the files will be saved
        self.create_directories()
        # Run the scf DFT calculation
        self.compute_scf()
        # Run the nscf DFT calculation
        self.compute_nscf()
        # Compute the Bloch phase
        self.compute_phase()
        # Save data from the DFT runs to be used by other programs
        self.save_data()

    def create_directories(self):
        # Create directory where data files will be saved
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        # Create directory where wavefunctions will be saved
        if not os.path.exists(self.wfc_dir):
            os.mkdir(self.wfc_dir)
        # Create directory where the Berry geometries will be saved
        if not os.path.exists(self.geometry_dir):
            os.mkdir(self.geometry_dir)

    # Save data to datafile.npy and other files for future use by the software
    def save_data(self):
        np.save(os.path.join(self.data_dir, "phase.npy"), self.phase)
        self.logger.debug(f"Saved phase.npy in {self.data_dir}")
        self.logger.info("\tPHASE saved to file phase.npy")

        np.save(os.path.join(self.data_dir, "neighbors.npy"), self.neigh)
        self.logger.debug(f"Saved neighbors.npy in {self.data_dir}")
        self.logger.info("\tNeighbors saved to file neighbors.npy")

        np.save(os.path.join(self.data_dir, "eigenvalues.npy"), self.eigenvalues)
        self.logger.debug(f"Saved eigenvalues.npy in {self.data_dir}")
        self.logger.info("\tEIGENVALUES saved to file eigenvalues.npy (Ry)")

        np.save(os.path.join(self.data_dir, "occupations.npy"), self.occupations)
        self.logger.debug(f"Saved occupations.npy in {self.data_dir}")
        self.logger.info("\tOCCUPATIONS saved to file occupations.npy")

        np.save(os.path.join(self.data_dir, "positions.npy"), self.rpoint)
        self.logger.debug(f"Saved positions.npy in {self.data_dir}")
        self.logger.info("\tPositions saved to file positions.npy (bohr)")

        np.save(os.path.join(self.data_dir, "kpoints.npy"), self.kpoints)
        self.logger.debug(f"Saved kpoints.npy in {self.data_dir}")
        self.logger.info("\tKPOINTS saved to file kpoints.npy (2pi/bohr)")

        np.save(os.path.join(self.data_dir, "nktoijl.npy"), self.nktijl)
        self.logger.debug(f"Saved nktoijl.npy in {self.data_dir}")
        self.logger.info("\tNKTOIJL saved to file nktoijl.npy, with convertion from nk to ijl")

        np.save(os.path.join(self.data_dir, "ijltonk.npy"), self.ijltonk)
        self.logger.debug(f"Saved ijltonk.npy in {self.data_dir}")
        self.logger.info("\tIJLTONK saved to file ijltonk.npy, with convertion from ijl to nk")

        with open("data/datafile.npy", "wb") as fich: #TODO: Try saving with np.savez
            np.save(fich, __version__)  # Version of berry where data was created
            np.save(fich, self.ref_name)  # Unique reference for the run
            np.save(fich,self.dimensions) # Number of dimensions of the material

            np.save(fich, self.work_dir)  # Working directory
            np.save(fich, self.data_dir)  # Directory for saving data
            np.save(fich, self.log_dir)  # Directory for the logs
            np.save(fich, self.geometry_dir)  # Directory for the Berry geometries
        
            np.save(fich, self.k0)  # Initial k-point
            np.save(fich, self.nkx)  # Number of k-points in the x direction
            np.save(fich, self.nky)  # Number of k-points in the y direction
            np.save(fich, self.nkz)  # Number of k-points in the z direction
            np.save(fich, self.__nks)  # Total number of k-points
            np.save(fich, self.step)  # Step between k-points
            np.save(fich, self.npr)  # Number of processors for the run
            np.save(fich, self.ref_point)  # Point in real space where all phases match

            np.save(fich, self.dft_dir)  # Directory of DFT files
            np.save(fich, self.scf)  # Name of scf file (without suffix)
            np.save(fich, self.nscf)  # Name of nscf file (without suffix)
            np.save(fich, self.prefix)  # Prefix of the DFT QE calculations
            np.save(fich, self.wfc_dir)  # Directory for the wfc files
            np.save(fich, self.out_dir)  # Directory for DFT saved files
            np.save(fich, self.dft_data_file)  # Path to DFT file with data of the run
            np.save(fich, self.program)  # DFT software to be used

            np.save(fich, self.a1)  # First lattice vector in real space
            np.save(fich, self.a2)  # Second lattice vector in real space
            np.save(fich, self.a3)  # Third lattice vector in real space
            np.save(fich, self.b1)  # First lattice vector in reciprocal space
            np.save(fich, self.b2)  # Second lattice vector in reciprocal space
            np.save(fich, self.b3)  # Third lattice vector in reciprocal space
            np.save(fich, self.nr1)  # Number of points of wfc in real space x direction
            np.save(fich, self.nr2)  # Number of points of wfc in real space y direction
            np.save(fich, self.nr3)  # Number of points of wfc in real space z direction
            np.save(fich, self.nr)  # Total number of points of wfc in real space
            np.save(fich, self.nbnd)  # Number of bands

            np.save(fich, self.non_colinear)  # If the calculation is noncolinear
            np.save(fich, self.lsda)  # Spin polarized calculation
            np.save(fich, self.nelec)  # Number of electrons
            np.save(fich, self.wfck2r)  # File for extracting DFT wfc to real space
            np.save(fich, self.vb)  # Valence band number
 
            np.save(fich, "dummy")  # Saving space for future values and compatibility
            np.save(fich, "dummy")  # Saving space for future values and compatibility
            np.save(fich, "dummy")  # Saving space for future values and compatibility
            np.save(fich, "dummy")  # Saving space for future values and compatibility
            np.save(fich, "dummy")  # Saving space for future values and compatibility
            np.save(fich, "dummy")  # Saving space for future values and compatibility
            np.save(fich, "dummy")  # Saving space for future values and compatibility
            np.save(fich, "dummy")  # Saving space for future values and compatibility
            np.save(fich, "dummy")  # Saving space for future values and compatibility
            np.save(fich, "dummy")  # Saving space for future values and compatibility

            self.logger.info("\tData saved to file datafile.npy")

        self.logger.footer()

    # Computes the phase e^i(k.r) for all pairs of k,r - points
    def compute_phase(self):
        self._extract_data_from_run()

        params = {
            "nr3": range(self.nr3),
            "nr2": range(self.nr2),
            "nr1": range(self.nr1),
        }

        with Pool(self.npr) as pool:
            self.rpoint = np.array(pool.starmap(self._compute_rpoints, product(*params.values())))

        self.phase = np.exp(1j * np.dot(self.rpoint, self.kpoints.T))

        self.neigh = self._compute_neighbors()

    # Creates array with the list of 1st neighbors for a 2D material   TODO: extend to 2nd neighbors and 3D
    def _compute_neighbors(self):
        nk = -1
        
        if self.dimensions == 1:
            neigh = np.full((self.__nks, 2), -1, dtype=np.int64)
            for i in range(self.nkx):  
                nk += 1
                if i == 0:
                    n0 = -1
                else:
                    n0 = nk - 1
                if i == self.nkx - 1:
                    n1 = -1
                else:
                    n1 = nk + 1 
                neigh[nk, :] = [n0,n1]
        elif self.dimensions == 2:
            neigh = np.full((self.__nks, 4), -1, dtype=np.int64)
            for j in range(self.nky):
                for i in range(self.nkx):
                    nk += 1
                    if i == 0:
                        n0 = -1
                    else:
                        n0 = nk - 1
                    if j == 0:
                        n1 = -1
                    else:
                        n1 = nk - self.nkx
                    if i == self.nkx - 1:
                        n2 = -1
                    else:
                        n2 = nk + 1
                    if j == self.nky - 1:
                        n3 = -1
                    else:
                        n3 = nk + self.nkx
                    neigh[nk, :] = [n0, n1, n2, n3]
        elif self.dimensions == 3:
            neigh = np.full((self.__nks, 6), -1, dtype=np.int64)
            for l in range(self.nkz):
                for j in range(self.nky):
                    for i in range(self.nkx):
                        nk += 1
                        if i == 0:
                            n0 = -1
                        else:
                            n0 = nk - 1
                        if j == 0:
                            n1 = -1
                        else:
                            n1 = nk - self.nkx
                        if i == self.nkx - 1:
                            n2 = -1
                        else:
                            n2 = nk + 1
                        if j == self.nky - 1:
                            n3 = -1
                        else:
                            n3 = nk + self.nkx
                        if l == 0:
                            n4 = -1
                        else:
                            n4 = nk - self.nkx*self.nky
                        if l == self.nkz - 1:
                            n5 = -1
                        else:
                            n5 = nk + self.nkx*self.nky
                        neigh[nk, :] = [n0, n1, n2, n3, n4, n5]
        else:
            self.logger.error(f"\tWrong number of dimensions: they can be only 1, 2 or 3.")
        return neigh

    # Runs the scf DFT calculation
    def compute_scf(self):
        # Establishes the name of the output file (assumes the original name ends in '.in')
        scf_out = self.scf[:-3] + ".out"
        if os.path.isfile(scf_out):
            self.logger.info(f"\t{os.path.basename(scf_out)} already exists. Skipping scf calculation.")
            return
        else:
            self.logger.info(f"\tRunning scf calculation.")

            command = f"{self.__mpi} pw.x -i {self.scf} > {scf_out}"
            os.system(command)
            self.logger.debug(f"\tRunning command: {command}")

    # Runs the nscf DFT calculation
    def compute_nscf(self):
        # Reads from template
        self._nscf_template()

        # Establishes the name of the output file (assumes the original name ends in '.in')
        nscf_out = self.nscf[:-3] + ".out"
        if os.path.isfile(nscf_out):
            self.logger.info(f"\t{os.path.basename(nscf_out)} already exists. Skipping nscf calculation.")
            # return
        else:
            self.logger.info(f"\tRunning nscf calculation.")

            command = f"{self.__mpi} pw.x -i {self.nscf} > {nscf_out}"
            os.system(command)
            self.logger.debug(f"Running command: {command}")

    # Makes a list of the points in real space
    def _compute_rpoints(self, l: int, k: int, i: int):
        return self.a1 * i / self.nr1 + self.a2 * k / self.nr2 + self.a3 * l / self.nr3

    # Reads the DFT data from the previous runs
    def _extract_data_from_run(self):
        self.logger.info(f"\n\tExtracting data from {self.dft_data_file}")

        root = ET.parse(self.dft_data_file).getroot()
        output = root.find("output")
        general = root.find("general_info")

        dft_version = float(general.find("creator").attrib["VERSION"][0:3])
        def_program = general.find("creator").attrib["NAME"]

        # Necessary due to different non backwards compatible versions of QE
        self.wfck2r = "wfck2r.oct" if dft_version >= 6.7 else "wfck2r.mat"

        self.logger.info(f"\n\tDFT {def_program} version {dft_version}.\n")

        self.logger.info(f"\n\tLattice vectors in units of a0 (bohr)")
        self.a1, self.a2, self.a3 = [np.array(list(map(float, it.text.split()))) for it in output.find("atomic_structure").find("cell")]
        self.logger.info("\t\ta1:", self.a1)
        self.logger.info("\t\ta2:", self.a2)
        self.logger.info("\t\ta3:", self.a3)

        self.logger.info(f"\n\tReciprocal lattice vectors in units of 2pi/a0 (2pi/bohr)")
        self.b1, self.b2, self.b3 = [np.array(list(map(float, it.text.split()))) for it in output.find("basis_set").find("reciprocal_lattice")]
        self.logger.info("\t\tb1:", self.b1)
        self.logger.info("\t\tb2:", self.b2)
        self.logger.info("\t\tb3:", self.b3)

        self.logger.info("\n\tNumber of points in real space in each direction")
        self.nr1 = int(output.find("basis_set").find("fft_grid").attrib["nr1"])
        self.nr2 = int(output.find("basis_set").find("fft_grid").attrib["nr2"])
        self.nr3 = int(output.find("basis_set").find("fft_grid").attrib["nr3"])
        self.nr = self.nr1 * self.nr2 * self.nr3
        self.logger.info(f"\t\tnr1: {self.nr1}")
        self.logger.info(f"\t\tnr2: {self.nr1}")
        self.logger.info(f"\t\tnr3: {self.nr1}")
        self.logger.info(f"\t\tnr: {self.nr}")
        self.ref_point = self.point * self.nr1 * self.nr2
        self.logger.info(f"\n\tPoint where phases match: {int(self.ref_point)}")

        nbnd = int(output.find("band_structure").find("nbnd").text)
        self.logger.info(f"\tNumber of bands in the DFT calculation: {nbnd}")
        self.nelec = float(output.find("band_structure").find("nelec").text)
        self.logger.info(f"\tNumber of electrons: {self.nelec}")
        nks = int(output.find("band_structure").find("nks").text)
        self.logger.info(f"\tNumber of k-points in the DFT calculation: {nks}")

        self.non_colinear = False if output.find("band_structure").find("noncolin").text == "false" else True
        self.logger.info(f"\tNon-colinear calculation: {self.non_colinear}")
        self.lsda = False if output.find("band_structure").find("lsda").text == "false" else True
        self.logger.info(f"\tSpin polarized calculation: {self.lsda}")

        self.vb = self.nelec - 1 if self.non_colinear or self.lsda else (self.nelec / 2) - 1
        if self.vb - int(self.vb) != 0:
            self.logger.warning(f"The system is a metal!")  # TODO: add supoort for metals
        self.vb = int(self.vb)
        self.logger.info(f"\tValence band is: {self.vb}\n")

        self.eigenvalues = 2 * np.array([list(map(float, it.text.split())) for it in output.find("band_structure").iter("eigenvalues")])
        # number 2 is to convert from hartree to rydberg energy units
        # QE uses Ha when saving eigenvalues in xml file, everywhere else uses Ry

        self.occupations = np.array([list(map(float, it.text.split())) for it in output.find("band_structure").iter("occupations")])

    # Creates the nscf input file based on the original scf input file
    def _nscf_template(self):
        with open(self.scf, "r") as f:
            scf = f.read()

        scf_w_nones = scf.replace("automatic", "tpiba").replace("scf", "nscf").split("\n")
        scf_w_nones = [line for line in scf_w_nones if line != ""]
        nscf_content = "\n".join(scf_w_nones[:-1])  # Remove scf kpoints

        # Replace nbnd with self.nbnd
        if re.search("nbnd", nscf_content):
            nscf_content = re.sub("nbnd.*", f"nbnd = {str(self.nbnd)},", nscf_content)
        elif re.search(r"SYSTEM\s*/", nscf_content):
            nscf_content = re.sub(r"SYSTEM\s*/", f"SYSTEM\n                        nbnd = {str(self.nbnd)},\n/", nscf_content)
        else:
            nscf_content = re.sub("SYSTEM", f"SYSTEM\n                        nbnd = {str(self.nbnd)},", nscf_content)

        # Guarantee no sym calculation for nscf
        if re.search("nosym", nscf_content):
            nscf_content = re.sub("nosym.*", f"nosym = .true.", nscf_content)
        else:
            nscf_content = re.sub("SYSTEM", f"SYSTEM\n                       nosym = .true.", nscf_content) 

        # Replace kpoints with self.nscf_kpoints
        nscf_content += f"\n{self.__nks}\n{self.nscf_kpoints}"

        # Write template to file
        with open(self.nscf, "w") as f:
            f.write(nscf_content)

    # Creates array of k-points where calculations will be performed
    def _build_kpoints(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Array of k-points
        kpoints = np.zeros((self.__nks, 3), dtype=np.float64)
        # Conversion from real k-point coordinates to integer coordinates
        nktoijl = np.zeros((self.__nks, 3), dtype=np.int64)
        # Conversion from integer coordinates to real k-point coordinates
        ijltonk = np.zeros((self.nkx, self.nky, self.nkz), dtype=np.int64)

        nk = 0
        for l in range(self.nkz):
            for j in range(self.nky):
                for i in range(self.nkx):
                    k1 = round(self.k0[0] + i * self.step, 8)
                    k2 = round(self.k0[1] + j * self.step, 8)
                    k3 = round(self.k0[2] + l * self.step, 8)

                    kkk = f"{k1:.7f}  {k2:.7f}  {k3:.7f}"
                    self.nscf_kpoints += kkk + "  1\n"
                    kpoints[nk, :] = [k1, k2, k3]
                    nktoijl[nk, :] = [i, j, l]
                    ijltonk[i, j, l] = nk
                    nk += 1

        return kpoints, nktoijl, ijltonk

    # Information for the log file
    def _log_inputs(self):
        self.logger.info("\tUsing python version: " + platform.python_version())

        self.logger.info("\n\tInputs:")
        self.logger.info(f"\tUnique reference name: {self.ref_name}")
        self.logger.info(f"\tStarting k-point of the mesh: {self.k0}")
        self.logger.info(f"\tNumber of k-points in the mesh: {self.nkx}x{self.nky}x{self.nkz}")
        self.logger.info(f"\tStep of the mesh: {self.step}")
        self.logger.info(f"\tTo calculate point in real space where all phases match: {self.point}")
        self.logger.info(f"\tNumber of bands to be calculated: {self.nbnd}\n")

        self.logger.info(f"\tWill use {self.npr} processors\n")
        self.logger.info(f"\tWorking directory: {self.work_dir}")
        self.logger.info(f"\tData directory: {self.data_dir}")
        self.logger.info(f"\tLog directory: {self.log_dir}")
        self.logger.info(f"\tWfc directory: {self.wfc_dir}")
        self.logger.info(f"\tGeometry directory: {self.geometry_dir}")

        self.logger.info(f"\n\tDFT directory: {self.dft_dir}")
        self.logger.info(f"\tDFT output directory: {self.out_dir}")
        self.logger.info(f"\tDFT pseudopotential directory: {self.pseudo_dir}")

        self.logger.info(f"\n\tscf input file: {self.scf}")
        self.logger.info(f"\tnscf input file: {self.nscf}")
        self.logger.info(f"\tDFT data file: {self.dft_data_file}\n")

