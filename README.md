# berry
<h2 align="center">
<img src="/docs/figures/BerryLogoBig.svg" width="300">
</h2><br>

**Berry** extracts the Bloch wavefunctions from DFT calculations in an ordered way so they can be directly used to make calculations.

It retrieves the wavefunctions and their gradients in reciprocal space totally ordered by unentangled bands, where continuity applies.

In particular, it calculates the Berry connections and curvatures from DFT calculations.

Then the Berry connections can be used to calculate the first order optical conductivity and the second order optical conductivity for second harmonic generation (SHG).

Therefore, this suite of programs can be used in calculations other than Berry connections and related topics, with small adaptations.

It still can only be used with 2D materials and with DFT suite Quantum Espresso.

It is expected that this software will evolve with many more possibilities in the near future.
The list of TODOs is already large.

- **Source code:** https://github.com/berry

