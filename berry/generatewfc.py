from typing import Optional

import os
import logging
import subprocess

import numpy as np

from berry import log

try:
    import berry._subroutines.loadmeta as m
    import berry._subroutines.loaddata as d
except:
    pass



def plot_psi(psi, x): # just for debugging
    import matplotlib.pyplot as plt
    cores = [
        "black",
        "blue",
        "green",
        "red",
        "grey",
        "brown",
        "violet",
        "seagreen",
        "dimgray",
        "darkorange",
        "royalblue",
        "darkviolet",
        "maroon",
        "yellowgreen",
        "peru",
        "steelblue",
        "crimson",
        "silver",
        "magenta",
        "yellow",
    ]
    fig = plt.figure(figsize=(6, 6))
    xarray = np.zeros((m.nr2, m.nr3))
    yarray = np.zeros((m.nr2, m.nr3))
    zarray = np.zeros((m.nr2, m.nr3))
    count = -1
    for j in range(m.nr3):
        for i in range(m.nr2):
            count = count + 1
            xarray[i, j] = d.r[x + count*m.nr1, 1]
            yarray[i, j] = d.r[x + count*m.nr1, 2]

    ax = fig.add_subplot(projection='3d')
    for banda in range(0, m.nbnd):
        count = -1
        for j in range(m.nr3):
            for i in range(m.nr2):
                count = count + 1
                zarray[i, j] = banda*3 + psi[x + count*m.nr1 + banda*m.nr]
        colorband = np.mod(banda,20)
        ax.plot_wireframe(xarray, yarray, zarray, color=cores[colorband])
    plt.show()


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


        self.cut = self._cut()

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

        if m.Deltaz != 0 and (m.dimensions == 2 or m.dimensions == 1):
            self.zcut = True
            self.z1 = m.z1
            if m.z1 + m.Deltaz > m.a3[2]:
                self.splitz = True
                self.z2 = m.z1 + m.Deltaz - m.a3[2]
            else:
                self.splitz = False
                self.z2 = m.z1 + m.Deltaz
        else:
            self.zcut = False
        if m.Deltay != 0 and m.dimensions == 1:
            self.ycut = True
            self.y1 = m.y1
            if m.y1 + m.Deltay > m.a2[1]:
                self.splity = True
                self.y2 = m.y1 + m.Deltay - m.a2[1]
            else:
                self.splity = False
                self.y2 = m.y1 + m.Deltay
        else:
            self.ycut = False

    def _cut(self):  # selection of the cut on the psi
        if self.zcut and self.ycut:
            if self.splitz and self.splity:
                def cut(psi):
                    for i in range(len(psi)):
                        if self.y1 < d.r[i % m.nr,1] or self.y2 > d.r[i % m.nr,1] or self.z1 < d.r[i % m.nr,2] or self.z2 > d.r[i % m.nr,2]:
                            psi[i] = 0
                    return psi
                
            elif self.splitz:
                def cut(psi):
                    for i in range(len(psi)):
                        if (self.y1 < d.r[i % m.nr,1] and self.y2 > d.r[i % m.nr,1]) or self.z1 < d.r[i % m.nr,2] or self.z2 > d.r[i % m.nr,2]:
                            psi[i] = 0
                    return psi
                
            elif self.splity:
                def cut(psi):
                    for i in range(len(psi)):
                        if self.y1 < d.r[i % m.nr,1] or self.y2 > d.r[i % m.nr,1] or (self.z1 < d.r[i % m.nr,2] and self.z2 > d.r[i % m.nr,2]):
                            psi[i] = 0
                    return psi
            else:
                def cut(psi):
                    for i in range(len(psi)):
                        if (self.y1 < d.r[i % m.nr,1] and self.y2 > d.r[i % m.nr,1]) or (self.z1 < d.r[i % m.nr,2] and self.z2 > d.r[i % m.nr,2]):
                            psi[i] = 0
                    return psi
        elif self.zcut:
            if self.splitz:
                def cut(psi):
                    for i in range(len(psi)):
                        if self.z1 < d.r[i % m.nr,2] or self.z2 > d.r[i % m.nr,2]:
                            psi[i] = 0
                    return psi
            else:
                def cut(psi):
                    for i in range(len(psi)):
                        if self.z1 < d.r[i % m.nr,2] and self.z2 > d.r[i % m.nr,2]:
                            psi[i] = 0
                    return psi
        elif self.ycut:
            if self.splity:
                def cut(psi):
                    for i in range(len(psi)):
                        if self.y1 < d.r[i % m.nr,1] or self.y2 > d.r[i % m.nr,1]:
                            psi[i] = 0
                    return psi
            else:
                def cut(psi):
                    for i in range(len(psi)):
                        if self.y1 < d.r[i % m.nr,1] and self.y2 > d.r[i % m.nr,1]:
                            psi[i] = 0
                    return psi
        else:
            cut = None
        return cut


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
            if self.cut is not None:
                psifinal0 = self.cut(psifinal0)
                psifinal1 = self.cut(psifinal1)
            # plot_psi(psifinal0, 1) # for debugging
            # plot_psi(psifinal1, 1) # for debugging

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
            if self.cut is not None:
                psifinal = self.cut(psifinal)
            psifinal = np.array(psifinal)

            # plot_psi(psifinal, 1) # for debugging
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
