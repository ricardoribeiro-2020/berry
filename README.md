# berry
This project extracts the Bloch wavefunctions from DFT calculations in an ordered way so they can be directly used to make calculations.

In practice it retrieves the wavefunctions and their gradients in reciprocal space totally ordered by unentangled bands, where continuity applies.

In particular, it calculates the Berry connections from DFT calculations.

Then the Berry connections can be used to calculate the first order optical conductivity and the second order optical conductivity for second harmonic generation (SHG).

Therefore, this suite of programs can be used in other calculations other than Berry connections and related topics, with small adaptations.

In this version 0.2 several improvements have been done related to the speed of the calculations and other optimizations.

It still can only be used with 2D materials and with DFT suite Quantum Espresso.

It is expected that this software will evolve with many more possibilities in the near future.

