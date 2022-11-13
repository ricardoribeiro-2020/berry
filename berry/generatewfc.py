from typing import Optional

import os
import logging
import subprocess

import numpy as np

from berry import log

try:
    import berry._subroutines.loaddata as d
except:
    pass


class WfcGenerator:
    def __init__(self, nk_points: Optional[int] = None , bands: Optional[int] = None, logger_name: str = "genwfc", logger_level: int = logging.INFO):
        if bands is not None and nk_points is None:
            raise ValueError("To generate a wavefunction for a single band, you must specify the k-point.")

        os.system("mkdir -p " + d.wfcdirectory)

        if nk_points is None:
            self.nk_points = range(d.nks)
        elif  bands is None:
            self.nk_points = nk_points
            self.bands = range(d.nbnd)
        else:
            self.nk_points = nk_points
            self.bands = bands
        self.logger = log(logger_name, "GENERATE WAVE FUNCTIONS", logger_level)

    def run(self):
        self.logger.header()
        if isinstance(self.nk_points, range):
            self.logger.info("Will run for all k-points and bands")
            self.logger.info(f"There are {d.nks} k-points and {d.nbnd} bands.\n")

            for nk in self.nk_points:
                self.logger.info(f"Calculating wfc for k-point {nk}")
                self._wfck2r(nk, 0, d.nbnd)
        else:
            if isinstance(self.bands, range):
                self.logger.info(f"Will run for k-point {self.nk_points} and all bands")
                self.logger.info(f"TThere are {d.nks} k-points and {d.nbnd} bands.\n")

                self.logger.info(f"\tCalculating wfc for k-point {self.nk_points}")
                self._wfck2r(self.nk_points, 0, d.nbnd)
            else:
                self.logger.info(f"Will run just for k-point {self.nk_points} and band {self.bands}.\n")
                self._wfck2r(self.nk_points, self.bands, 1)
        self.logger.footer()

    def _wfck2r(self, nk_point: int, initial_band: int, number_of_bands: int):
        shell_cmd = self._get_command(nk_point, initial_band, number_of_bands)

        output = subprocess.check_output(shell_cmd, shell=True)

        out1 = (output.decode("utf-8")
                .replace(")", "j")
                .replace(", -", "-")
                .replace(",  ", "+")
                .replace("(", "")
                )

        # puts the wavefunctions into a numpy array
        psi = np.fromstring(out1, dtype=complex, sep="\n")

        # For each band, find the value of the wfc at the specific point rpoint (in real space)
        psi_rpoint = np.array([psi[int(d.rpoint) + d.nr * i] for i in range(number_of_bands)])

        # Calculate the phase at rpoint for all the bands
        deltaphase = np.arctan2(psi_rpoint.imag, psi_rpoint.real)

        # and the modulus
        mod_rpoint = np.absolute(psi_rpoint)

        psifinal = []
        for i in range(number_of_bands):
            self.logger.debug(f"\t{nk_point:6d}  {(i + initial_band):4d}  {mod_rpoint[i]:12.8f}  {deltaphase[i]:12.8f}   {not mod_rpoint[i] < 1e-5}")
            
            # Subtract the reference phase for each point
            psifinal += list(psi[i * d.nr : (i + 1) * d.nr] * np.exp(-1j * deltaphase[i]))
        psifinal = np.array(psifinal)

        outfiles = map(lambda band: os.path.join(d.wfcdirectory, f"k0{nk_point}b0{band+initial_band}.wfc"), range(number_of_bands))
        for i, outfile in enumerate(outfiles):
            with open(outfile, "wb") as fich:
                np.save(fich, psifinal[i * d.nr : (i + 1) * d.nr])

    def _get_command(self, nk_point: int, initial_band: int, number_of_bands: int):
        mpi = "" if d.npr == 1 else f"mpirun -np {d.npr} "
        command =f"&inputpp prefix = '{d.prefix}',\
                        outdir = '{d.outdir}',\
                        first_k = {nk_point + 1},\
                        last_k = {nk_point + 1},\
                        first_band = {initial_band + 1},\
                        last_band = {initial_band + number_of_bands},\
                        loctave = .true., /"
        return f'echo "{command}" | {mpi} wfck2r.x > tmp; tail -{d.nr * number_of_bands} {d.wfck2r}'