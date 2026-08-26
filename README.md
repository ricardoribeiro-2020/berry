# berry
<h2 align="center">
<img src="/docs/figures/BerryLogoBig.svg" width="300">
</h2><br>

**Berry** extracts the Bloch wavefunctions from DFT calculations in an ordered way so they can be directly used to make calculations.

It retrieves the wavefunctions and their gradients in reciprocal space totally ordered by unentangled bands, where continuity applies.

In particular, it calculates the Berry connections and curvatures from DFT calculations.

Then the Berry connections can be used to calculate the first order optical conductivity and the second order optical conductivity for second harmonic generation (SHG).

Therefore, this suite of programs can be used in calculations other than Berry connections and related topics, with small adaptations.

Version 2 can be used with 1D, 2D and 3D materials, with either no spin or noncollinear DFT calculations.
It uses DFT suite Quantum Espresso.

It is expected that this software will evolve with many more possibilities in the near future.
The list of TODOs is already large.

- **Source code:** https://github.com/ricardoribeiro-2020/berry

- **Aknowledgement:** We aknowledge the Fundação para a Ciência e a Tecnologia (FCT)
under project  **QUEST2D - Excitations in quantum 2D materials**
PTDC/FIS-MAC/2045/2021 and in the framework of the Strategic Funding UIDB/04650/2020.

## Installation

```
pip install berry-suite
```

or, from a source checkout:

```
pip install -e .
```

A working [Quantum ESPRESSO](https://www.quantum-espresso.org/) installation
(version 7.5 or higher) must be present and on your `$PATH`.
Noncollinear calculations additionally require a patched `wfck2rFR.x`.
See [docs/INSTALL](docs/INSTALL) for details.

## Usage

Installing puts two commands on your `$PATH`: `berry`, which runs the
calculation pipeline, and `berry-vis`, which draws the results.

```
berry [package options] script parameter [script options]
```

The scripts must be run **in this order**, because each one uses the results of
the previous ones:

| Command | What it does |
| --- | --- |
| `preprocess` | Run and extract data from DFT calculations. This should be the first script to run. |
| `wfcgen` | Extracts wavefunctions from DFT calculations. |
| `degenrot` | Finds degenerate subspaces and rotates wavefunctions into a smooth basis. |
| `dot` | Calculates the dot product of Bloch factors of nearby wavefunctions. |
| `cluster` | Classifies the eigenstates in bands. (`cluster0` is a backwards-compatible alias.) |
| `basis` | Rotates degenerate wavefunctions flagged by the clustering into a smooth basis. |
| `r2k` | Converts wavefunctions from r-space to k-space. |
| `geometry` | Calculates the Berry connections and curvatures. |
| `conductivity` | Calculates the optical linear conductivity of the system. |
| `shg` | Calculates the second harmonic generation conductivity of the system. |

`berry-vis` provides `debug`, `geometry` and `bands` for inspecting the data,
drawing the Berry connection and curvature vectors, and showing the electronic
band structure.

Run `berry --help` or `berry <script> --help` for the options of each script,
and see [docs/documentation.pdf](docs/documentation.pdf) for the full manual.

Copyright (c) 2022, 2023

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

