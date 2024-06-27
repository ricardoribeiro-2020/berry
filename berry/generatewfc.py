from typing import Optional

import os
import logging
import subprocess

from berry._subroutines.loadmeta import Deltay, Deltaz
import numpy as np

from berry import log

try:
    import berry._subroutines.loadmeta as m
    import berry._subroutines.loaddata as d
except:
    pass


class WfcGenerator:
    def __init__(self,
                 nk_points: Optional[int] = None ,
                 bands: Optional[int] = None,
                 logger_name: str = "genwfc",
                 logger_level: int = logging.INFO,
                 flush: bool = False
                ):

        if bands is not None and nk_points is None:
            raise ValueError("To generate a wavefunction for a single band, you must specify the k-point.")

        os.system("mkdir -p " + m.wfcdirectory)

        if nk_points is None and bands is None:
            self.nk_points = range(m.nks)
            self.bands = range(m.wfcut + 1, m.nbnd)
        elif  bands is None:                          # Just for debugging
            self.nk_points = nk_points
            self.bands = range(m.wfcut + 1, m.nbnd)
        elif nk_points is None:                       # Just for debugging
            self.nk_points = range(m.nks)
            self.bands = bands
        else:                                         # Just for debugging
            self.nk_points = nk_points
            self.bands = bands
        self.ref_name = m.refname
        self.logger = log(logger_name, "GENERATE WAVE FUNCTIONS", level=logger_level, flush=flush)


    def run(self):
        # prints header on the log file
        self.logger.header()

        # Logs the parameters for the run
        self._log_run_params(m.initial_band, m.number_of_bands, m.final_band)

        # Sets the program used for converting wavefunctions to the real space
        if m.noncolin:
            self.k2r_program = "wfck2rFR.x"
            self.logger.info("\tNoncolinear calculation, will use wfck2rFR.x")
        else:
            self.k2r_program = "wfck2r.x"
            self.logger.info("\tNonrelativistic calculation, will use wfck2r.x")


        # Set which k-points and bands will use (for debugging)
        if isinstance(self.nk_points, range):
            self.logger.info("\n\tWill run for all k-points and bands")
            self.logger.info(f"\tThere are {m.nks} k-points and {m.number_of_bands} bands.\n")

            for nk in self.nk_points:
                self.logger.info(f"\tCalculating wfc for k-point {nk}")
                self._wfck2r(nk, m.initial_band, m.final_band, m.number_of_bands)
        else:                       # Just for debugging
            if isinstance(self.bands, range):
                self.logger.info(f"\tWill run for k-point {self.nk_points} and all bands")
                self.logger.info(f"\tThere are {m.nks} k-points and {m.number_of_bands} bands.\n")

                self.logger.info(f"\tCalculating wfc for k-point {self.nk_points}")
                self._wfck2r(self.nk_points, m.initial_band, m.final_band, m.number_of_bands)
            else:
                self.logger.info(f"\tWill run just for k-point {self.nk_points} and band {self.bands}.\n")
                self._wfck2r(self.nk_points, self.bands, 1, 1)

        self.logger.info("\n\tRemoving temporary file 'tmp'")
        os.system(f"rm {os.getcwd()}/tmp")
        self.logger.info(f"\tRemoving quantum expresso output file '{m.wfck2r}'")
        os.system(f"rm {os.path.join(os.getcwd(),m.wfck2r)}")

        self.logger.footer()


    def _log_run_params(self, initial_band, number_of_bands, final_band):
        self.logger.info(f"\tUnique reference of run: {self.ref_name}")
        self.logger.info(f"\tWavefunctions will be saved in directory {m.wfcdirectory}")
        self.logger.info(f"\tDFT files are in directory {m.dftdirectory}")
        self.logger.info(f"\tThis program will run in {m.npr} processors\n")

        self.logger.info(f"\tTotal number of k-points: {m.nks}")
        self.logger.info(f"\tNumber of r-points in each direction: {m.nr1} {m.nr2} {m.nr3}")
        self.logger.info(f"\tTotal number of points in real space: {m.nr}")
        self.logger.info(f"\tWill use bands from {initial_band} to {final_band}")
        self.logger.info(f"\tTotal number of bands to be used: {number_of_bands}\n")

        self.logger.info(f"\tPoint choosen for sincronizing phases:  {m.rpoint}\n")

        if m.z1 != None and (m.dimensions == 2 or m.dimensions == 1):
            zcut = True
            z1 = m.z1
            if m.z1 + m. Deltaz > m.a3:
                splitz = True
                z2 = m.z1 + m.Deltaz - m.a3
            else:
                splitz = False
                z2 = m.z1 + m.Deltaz
        else:
            zcut = False
        if m.y1 != None and m.dimensions == 1:
            ycut = True
            y1 = m.y1
            if m.y1 + m.Deltay > m.a2:
                splity = True
                y2 = m.y1 + m.Deltay - m.a2
            else:
                splity = False
                y2 = m.y1 + m.Deltay
        else:
            ycut = False


    def _wfck2r(self, nk_point: int, initial_band: int, final_band: int, number_of_bands: int):
        # Set the command to run
        shell_cmd = self._get_command(nk_point, initial_band, final_band, number_of_bands)

        # Runs the command
        output = subprocess.check_output(shell_cmd, shell=True)

        # Converts fortran complex numbers to numpy format
        out1 = (output.decode("utf-8")
                .replace(")", "j")
                .replace(", -", "-")
                .replace(",  ", "+")
                .replace("(", "")
                )




        if m.noncolin:
            # puts the wavefunctions into a numpy array
            psi = np.fromstring(out1, dtype=complex, sep="\n")

            # For each band, find the value of the wfc at the specific point rpoint (in real space)
            psi_rpoint = np.array([psi[int(m.rpoint) + m.nr * i] for i in range(0,2*number_of_bands,2)])

            # Calculate the phase at rpoint for all the bands
            deltaphase = np.arctan2(psi_rpoint.imag, psi_rpoint.real)

            # and the modulus of the wavefunction at the reference point rpoint (
            # will be used to verify if the wavefunction at rpoint is significantly different from zero)
            mod_rpoint = np.absolute(psi_rpoint)

            psifinal0, psifinal1 = [], []

            for i in range(0,2*number_of_bands,2):
                self.logger.debug(f"\t{nk_point:6d}  {(int(i/2) + initial_band):4d}  {mod_rpoint[int(i/2)]:12.8f}  {deltaphase[int(i/2)]:12.8f}   {not mod_rpoint[int(i/2)] < 1e-5}")

                # Subtract the reference phase for each point
                psifinal0 += list(psi[i * m.nr : (i + 1) * m.nr] * np.exp(-1j * deltaphase[int(i/2)]))                # first part of spinor, all bands
                psifinal1 += list(psi[m.nr + i * m.nr : m.nr + (i + 1) * m.nr] * np.exp(-1j * deltaphase[int(i/2)]))  # second part of spinor, all bands

            outfiles0 = map(lambda band: os.path.join(m.wfcdirectory, f"k0{nk_point}b0{band+initial_band}-0.wfc"), range(number_of_bands))
            outfiles1 = map(lambda band: os.path.join(m.wfcdirectory, f"k0{nk_point}b0{band+initial_band}-1.wfc"), range(number_of_bands))

            for i, outfile in enumerate(outfiles0):
                with open(outfile, "wb") as fich:
                    np.save(fich, psifinal0[i * m.nr : (i + 1) * m.nr])
            for i, outfile in enumerate(outfiles1):
                with open(outfile, "wb") as fich:
                    np.save(fich, psifinal1[i * m.nr : (i + 1) * m.nr])

        else:
            # puts the wavefunctions into a numpy array
            psi = np.fromstring(out1, dtype=complex, sep="\n")

            # For each band, find the value of the wfc at the specific point rpoint (in real space)
            psi_rpoint = np.array([psi[int(m.rpoint) + m.nr * i] for i in range(number_of_bands)])

            # Calculate the phase at rpoint for all the bands
            deltaphase = np.arctan2(psi_rpoint.imag, psi_rpoint.real)

            # and the modulus of the wavefunction at the reference point rpoint (
            # will be used to verify if the wavefunction at rpoint is significantly different from zero)
            mod_rpoint = np.absolute(psi_rpoint)

            psifinal = []
            for i in range(number_of_bands):
                self.logger.debug(f"\t{nk_point:6d}  {(i + initial_band):4d}  {mod_rpoint[i]:12.8f}  {deltaphase[i]:12.8f}   {not mod_rpoint[i] < 1e-5}")

                # Subtract the reference phase for each point
                psifinal += list(psi[i * m.nr : (i + 1) * m.nr] * np.exp(-1j * deltaphase[i]))
            psifinal = np.array(psifinal)

            outfiles = map(lambda band: os.path.join(m.wfcdirectory, f"k0{nk_point}b0{band+initial_band}.wfc"), range(number_of_bands))
            for i, outfile in enumerate(outfiles):
                with open(outfile, "wb") as fich:
                    np.save(fich, psifinal[i * m.nr : (i + 1) * m.nr])

    def _get_command(self, nk_point: int, initial_band: int, final_band: int, number_of_bands: int):
        mpi = "" if m.npr == 1 else f"mpirun -np {m.npr} "
        command =f"&inputpp prefix = '{m.prefix}',\
                        outdir = '{m.outdir}',\
                        first_k = {nk_point + 1},\
                        last_k = {nk_point + 1},\
                        first_band = {initial_band + 1},\
                        last_band = {final_band + 1},\
                        loctave = .true., /"
        if m.noncolin:
            return f'echo "{command}" | {mpi} wfck2rFR.x > tmp; tail -{m.nr * number_of_bands*2} {m.wfck2r}'
        else:
            return f'echo "{command}" | {mpi} wfck2r.x > tmp; tail -{m.nr * number_of_bands} {m.wfck2r}'