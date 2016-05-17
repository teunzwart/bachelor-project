"""Performance critical functions for Ising Model Monte Carlo simulations."""

import numpy as np
import cython
import cython.parallel
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_lattice_energy(np.ndarray[np.int_t, ndim=2] lattice, int lattice_size, int bond_energy):
    """
    Calculate the energy of the lattice using the Ising model Hamiltonian in zero-field.

    The Cython implementations is 100 to 500 times faster than the pure Python implementation.
    (1.64 ms vs 660 ms for 1024**2 lattice)
    To increase performance only the bonds between x and x + 1 for the same y and
    y and y + 1 for the same x are calculated.
    Plain function: 1 loop, best of 3: 669 ms per loop
    Class method: 1 loop, best of 3: 760 ms per loop
    Simple Cython: 1 loop, best of 3: 500 ms per loop
    Some typing: 1 loop, best of 3: 420 ms per loop
    Full cython solution: 100 loops, best of 3: 3.57 ms per loop
    Including parallel execution: 100 loops, best of 3: 1.64 ms per loop Both plain and cython versions give same output, 395 times speedup
    """
    cdef int energy = 0
    cdef int y, x, center, offset_y, offset_x, xnn, ynn
    with nogil, cython.parallel.parallel(num_threads=4):
        for y in cython.parallel.prange(lattice_size):
            # The same for all x values, so can be precalculated.
            offset_y = y + 1
            # Wraparound the lattice. Note that truncated assignment operaters are used
            # to infer reduction variables, so you can't use those for offset_y.
            if y + 1 >= lattice_size:
                offset_y = offset_y - lattice_size
            for x in range(lattice_size):
                offset_x = x + 1
                if x + 1 >= lattice_size:
                    offset_x = offset_x - lattice_size
                center = lattice[y, x]
                # For some reason this is faster as a typed variable than directly
                # putting those lattice lookups in the if-statements.
                xnn = lattice[y, offset_x]
                ynn = lattice[offset_y, x]
                if xnn == center:
                    # Only one reduction operator can be used, so substraction has to be done this way.
                    energy += -1 * bond_energy
                else:
                    energy += bond_energy
                if ynn == center:
                    energy += -1 * bond_energy
                else:
                    energy += bond_energy
    return energy
